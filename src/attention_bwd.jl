@kernel cpu=false inbounds=true function _flash_attention_bwd!(
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
    ::Val{emb_dim}, ::Val{q_seq_tiles}, ::Val{kv_seq_tiles},
) where {T, emb_dim, q_seq_tiles, kv_seq_tiles}
    gsz = prod(@groupsize())

    # TODO:
    # - use T for q, k if we have enough shmem
    # NOTE: q, k shmem are in FP16 to fit into 64 KiB budget on Navi 3.
    q_shm = @localmem Float16 (gsz, emb_dim)
    k_shm = @localmem Float16 (emb_dim, gsz)
    s_shm = @localmem T (gsz, gsz)
    Δ_shm = @localmem T (emb_dim, gsz)
    d_shm = @localmem T (emb_dim, gsz)

    tidx = @index(Local)
    gidx = @index(Group, NTuple)

    for start_n in 1:kv_seq_tiles
        lo = (start_n - 1) * gsz
        # TODO use when causal
        # q_offset = lo
        q_offset = 0

        for i in 1:emb_dim
            k_shm[i, tidx] = k[i, tidx + lo, gidx[1], gidx[2]]
        end

        # TODO use when causal
        # for start_m in start_n:kv_seq_tiles
        for start_m in 1:q_seq_tiles
            lo_inner = (start_m - 1) * gsz

            for i in 1:emb_dim
                Δ_shm[i, tidx] = Δ[i, tidx + q_offset, gidx[1], gidx[2]]
            end
            for i in 1:emb_dim
                q_shm[tidx, i] = q[i, tidx + q_offset, gidx[1], gidx[2]]
            end
            @synchronize()

            # recompute q' * k (L_q, L_k)
            mma!(s_shm, q_shm, k_shm, cfg, tidx, mma_non_acc_fn)
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
            mma!(d_shm, Δ_shm, s_shm, cfg_dv, tidx, mma_acc_fn)
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
            mma!(d_shm, s_shm, q_shm, cfg_dk, tidx, mma_acc_fn)
            @synchronize()
            for i in 1:emb_dim
                dk[i, tidx + lo, gidx[1], gidx[2]] = d_shm[i, tidx]
            end

            # compute dq = dot(ds, k) (L_q, L_k) * (L_k, emb_dim)
            for i in 1:emb_dim
                d_shm[i, tidx] = dq[i, tidx + lo_inner, gidx[1], gidx[2]]
            end
            @synchronize()
            mma!(d_shm, s_shm, k_shm, cfg_dq, tidx, mma_acc_fn)
            @synchronize()
            for i in 1:emb_dim
                dq[i, tidx + lo_inner, gidx[1], gidx[2]] = d_shm[i, tidx]
            end

            q_offset += gsz
        end
    end
end

@kernel cpu=false inbounds=true function _flash_attention_bwd_preprocess!(
    # Output.
    Δ_scaled::AbstractArray{T, 4},
    δ::AbstractArray{T, 3},
    # Input.
    Δ::AbstractArray{T, 4},
    o::AbstractArray{T, 4},
    ls::AbstractArray{T, 3},
    ::Val{emb_dim},
) where {T, emb_dim}
    gsz = prod(@groupsize())

    tidx = @index(Local)
    gidx = @index(Group, NTuple)
    q_offset = (gidx[1] - 1) * gsz

    # Δ = Δ / ls
    inv_denom = inv(ls[tidx + q_offset, gidx[2], gidx[3]])
    Δ_scaled_v = @view(Δ_scaled[:, tidx + q_offset, gidx[2], gidx[3]])
    Δ_v = @view(Δ[:, tidx + q_offset, gidx[2], gidx[3]])
    for i in 1:emb_dim
        Δ_scaled_v[i] = Δ_v[i] * inv_denom
    end

    # δ = sum(o * do; dims=2) # dims=2 in the (B, H, L, E) format
    o_v = @view(o[:, tidx + q_offset, gidx[2], gidx[3]])
    d = zero(T)
    for i in 1:emb_dim
        d += Δ_scaled_v[i] * o_v[i]
    end
    δ[tidx + q_offset, gidx[2], gidx[3]] = d
end


function ∇flash_attention(
    Δ::AbstractArray{T, 4},
    o::AbstractArray{T, 4}, ms::AbstractArray{T, 3}, ls::AbstractArray{T, 3},
    q::AbstractArray{T, 4}, k::AbstractArray{T, 4}, v::AbstractArray{T, 4},
) where T
    emb_dim, QL, H, N = size(q)
    KL = size(k, 2)
    @assert size(k) == size(v)
    gsz = flash_attention_groupsize(T; emb_dim, target_shmem=64 * 1024) # TODO query available shmem
    @assert gsz ≤ QL

    q_seq_tiles = cld(QL, gsz)
    kv_seq_tiles = cld(KL, gsz)
    threads = (gsz, 1, 1)
    ndrange = (gsz * q_seq_tiles, H, N)

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
    TM, TN = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg = FATileConfig{BM, BK, BN, TM, TN, false, false, false}

    # mma config for dv = Δ * s_shm tile: (emb_dim, L_q) * s_shm(L_q, L_k).
    BM, BK, BN = emb_dim, gsz, gsz
    TM, TN = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg_dv = FATileConfig{BM, BK, BN, TM, TN, false, false, false}

    # mma config for dk
    BM, BK, BN = gsz, gsz, emb_dim
    TM, TN = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg_dk = FATileConfig{BM, BK, BN, TM, TN, true, false, true}

    # mma config for dq
    BM, BK, BN = gsz, gsz, emb_dim
    TM, TN = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg_dq = FATileConfig{BM, BK, BN, TM, TN, false, true, true}

    # mma config for ds
    BM, BK, BN = gsz, emb_dim, gsz
    TM, TN = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg_ds = FATileConfig{BM, BK, BN, TM, TN, true, false, false}

    _flash_attention_bwd!(kab, threads)(
        cfg, cfg_dv, cfg_dk, cfg_dq, cfg_ds,
        # Output.
        dq, dk, dv,
        # Input.
        Δ_scaled, δ,
        o, ms, ls,
        q, k, v,
        Val(emb_dim), Val(q_seq_tiles), Val(kv_seq_tiles); ndrange)

    return dq, dk, dv
end
