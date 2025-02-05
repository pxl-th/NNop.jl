@kernel cpu=false inbounds=true function _flash_attention_bwd_v2!(
    cfg, cfg_dv, cfg_dk, cfg_dq, cfg_ds,
    # Output.
    dq::AbstractArray{T, 4}, dk::AbstractArray{T, 4}, dv::AbstractArray{T, 4},
    # Input.
    Δ::AbstractArray{T, 4},
    δ::AbstractArray{T, 3},
    o::AbstractArray{T, 4},
    ms::AbstractArray{T, 3},
    ls::AbstractArray{T, 3},
    q::AbstractArray{T, 4}, k::AbstractArray{T, 4}, v::AbstractArray{T, 4},
    ::Val{emb_dim}, ::Val{n_seq_tiles},
) where {T, emb_dim, n_seq_tiles}
    gsz = prod(@groupsize())

    # TODO use T if it is not FP32
    # NOTE: q, k shmem are in FP16 to fit into 64 KiB budget.
    q_shm = @localmem Float16 (gsz, emb_dim)
    k_shm = @localmem Float16 (emb_dim, gsz)
    s_shm = @localmem T (gsz, gsz)
    Δ_shm = @localmem T (emb_dim, gsz)
    d_shm = @localmem T (emb_dim, gsz)

    tidx = @index(Local)
    gidx = @index(Group, NTuple)

    for start_n in 1:n_seq_tiles
        lo = (start_n - 1) * gsz
        # TODO use when causal
        # q_offset = lo
        q_offset = 0

        for i in 1:emb_dim
            k_shm[i, tidx] = k[i, tidx + lo, gidx[1], gidx[2]]
        end

        # TODO use when causal
        # for start_m in start_n:n_seq_tiles
        for start_m in 1:n_seq_tiles
            lo_inner = (start_m - 1) * gsz

            for i in 1:emb_dim
                Δ_shm[i, tidx] = Δ[i, tidx + q_offset, gidx[1], gidx[2]]
            end
            for i in 1:emb_dim
                q_shm[tidx, i] = q[i, tidx + q_offset, gidx[1], gidx[2]]
            end
            @synchronize()

            # recompute q' * k (L_q, L_k)
            mma!(s_shm, q_shm, k_shm, cfg, tidx, false)
            @synchronize()

            # recompute softmax dims=2
            m_i = ms[tidx + lo_inner, gidx[1], gidx[2]]
            for i in 1:gsz
                s_shm[tidx, i] = exp(s_shm[tidx, i] - m_i)
            end
            @synchronize() # TODO remove

            # compute dv += Δ (emb_dim, L_q) * s_shm (L_q, L_k) = (emb_dim, L_k)
            for i in 1:emb_dim
                d_shm[i, tidx] = dv[i, tidx + lo, gidx[1], gidx[2]]
            end
            @synchronize()
            mma!(d_shm, Δ_shm, s_shm, cfg_dv, tidx, true)
            @synchronize()
            for i in 1:emb_dim
                dv[i, tidx + lo, gidx[1], gidx[2]] = d_shm[i, tidx]
            end

            # compute:
            # - d = Δ' * v: (L_q, emb_dim) * (emb_dim, L_k) = (L_q, L_k)
            # - ((0 - d_i) .+ d) .* s_shm
            for i in 1:emb_dim
                d_shm[i, tidx] = v[i, tidx + lo, gidx[1], gidx[2]]
            end
            @synchronize()
            mma!(
                s_shm, Δ_shm, d_shm, cfg_ds, tidx,
                (out, res, x, y) -> begin
                    d_i = δ[x + lo_inner, gidx[1], gidx[2]]
                    out * (res - d_i)
                end,
            )

            # dk = s_shm' * q_shm: (L_k, L_q) * (L_q, emb_dim) -> (L_k, emb_dim) -> transpose into shmem
            for i in 1:emb_dim
                d_shm[i, tidx] = dk[i, tidx + lo, gidx[1], gidx[2]]
            end
            @synchronize()
            mma!(d_shm, s_shm, q_shm, cfg_dk, tidx, true)
            @synchronize()
            for i in 1:emb_dim
                dk[i, tidx + lo, gidx[1], gidx[2]] = d_shm[i, tidx]
            end

            # compute dq = dot(ds, k) (L_q, L_k) * (L_k, emb_dim)
            for i in 1:emb_dim
                d_shm[i, tidx] = dq[i, tidx + lo_inner, gidx[1], gidx[2]]
            end
            @synchronize()
            mma!(d_shm, s_shm, k_shm, cfg_dq, tidx, true)
            @synchronize()
            for i in 1:emb_dim
                dq[i, tidx + lo_inner, gidx[1], gidx[2]] = d_shm[i, tidx]
            end

            q_offset += gsz
        end
    end
end


function ∇flash_attention_v2(
    Δ::AbstractArray{T, 4},
    o::AbstractArray{T, 4}, ms::AbstractArray{T, 3}, ls::AbstractArray{T, 3},
    q::AbstractArray{T, 4}, k::AbstractArray{T, 4}, v::AbstractArray{T, 4},
) where T
    emb_dim, L, H, N = size(q)

    gsz = 64
    @assert L ≥ gsz
    n_seq_tiles = cld(L, gsz)
    threads = (gsz, 1, 1)
    ndrange = (gsz * n_seq_tiles, H, N)

    kab = get_backend(q)
    Δ_scaled = similar(Δ)
    δ = similar(ls)
    _flash_attention_bwd_preprocess!(kab, threads)(
        # Output.
        Δ_scaled, δ,
        # Input.
        Δ, o, ls,
        Val(emb_dim); ndrange)

    dq = KA.zeros(kab, T, size(q))
    dk = KA.zeros(kab, T, size(k))
    dv = KA.zeros(kab, T, size(v))
    threads = (gsz, 1)
    ndrange = (gsz * H, N)

    # mma config for Q' * K tile: (L_q, emb_dim) * (emb_dim, L_k).
    BM, BK, BN = gsz, emb_dim, gsz
    TM, TN = 8, 8
    cfg = FATileConfig{BM, BK, BN, TM, TN, false, false, false}
    @assert prod(threads) == (BM * BN) ÷ (TM * TN)

    # mma config for dv = Δ * s_shm tile: (emb_dim, L_q) * s_shm(L_q, L_k).
    BM, BK, BN = emb_dim, gsz, gsz
    TM, TN = 8, 8
    cfg_dv = FATileConfig{BM, BK, BN, TM, TN, false, false, false}
    @assert prod(threads) == (BM * BN) ÷ (TM * TN)

    # mma config for dk
    BM, BK, BN = gsz, gsz, emb_dim
    TM, TN = 8, 8
    cfg_dk = FATileConfig{BM, BK, BN, TM, TN, true, false, true}
    @assert prod(threads) == (BM * BN) ÷ (TM * TN)

    # mma config for dq
    BM, BK, BN = gsz, gsz, emb_dim
    TM, TN = 8, 8
    cfg_dq = FATileConfig{BM, BK, BN, TM, TN, false, true, true}
    @assert prod(threads) == (BM * BN) ÷ (TM * TN)

    # mma config for ds
    BM, BK, BN = gsz, emb_dim, gsz
    TM, TN = 8, 8
    cfg_ds = FATileConfig{BM, BK, BN, TM, TN, true, false, false}
    @assert prod(threads) == (BM * BN) ÷ (TM * TN)

    _flash_attention_bwd_v2!(kab, threads)(
        cfg, cfg_dv, cfg_dk, cfg_dq, cfg_ds,
        # Output.
        dq, dk, dv,
        # Input.
        Δ_scaled, δ,
        o, ms, ls,
        q, k, v,
        Val(emb_dim), Val(n_seq_tiles); ndrange)

    return dq, dk, dv
end

