@kernel unsafe_indices=true cpu=false inbounds=true function _flash_attention_bwd!(
    cfg, cfg_dv, cfg_dk, cfg_dq, cfg_ds,
    # Output.
    dq::AbstractArray{T, 4}, dk::AbstractArray{T, 4}, dv::AbstractArray{T, 4},
    # Input.
    Δ::AbstractArray{T, 4},
    δ::AbstractArray{T, 3},
    o::AbstractArray{T, 4},
    ms::AbstractArray{T, 3},
    q::AbstractArray{T, 4}, k::AbstractArray{T, 4}, v::AbstractArray{T, 4},
    scale::T,
    ::Val{emb_dim}, ::Val{q_seq_tiles}, ::Val{kv_seq_tiles}, ::Val{in_seq_bounds},
) where {T, emb_dim, q_seq_tiles, kv_seq_tiles, in_seq_bounds}
    gsz = @groupsize()[1]

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

    @inline function sh_load_emb!(dest, source, offset, mask::Bool, ::Val{transposed}) where transposed
        @unroll for i in 1:emb_dim
            x, y = transposed ? (tidx, i) : (i, tidx)
            @inbounds dest[x, y] = mask ? source[i, tidx + offset, gidx[1], gidx[2]] : zero(T)
        end
    end

    for start_n in 1:kv_seq_tiles
        lo = (start_n - 1) * gsz
        # TODO use when causal
        # q_offset = lo
        q_offset = 0

        in_k_seq_bounds = in_seq_bounds || tidx + lo ≤ size(k, 2)
        sh_load_emb!(k_shm, k, lo, in_k_seq_bounds, Val{false}())

        # TODO use when causal
        # for start_m in start_n:kv_seq_tiles
        for start_m in 1:q_seq_tiles
            lo_inner = (start_m - 1) * gsz

            in_q_seq_bounds = in_seq_bounds || tidx + q_offset ≤ size(q, 2)
            sh_load_emb!(Δ_shm, Δ, q_offset, in_q_seq_bounds, Val{false}())
            sh_load_emb!(q_shm, q, q_offset, in_q_seq_bounds, Val{true}())
            @synchronize()

            # recompute q' * k (L_q, L_k)
            mma!(s_shm, q_shm, k_shm, cfg, tidx, (res, c_shm, x, y) -> res * scale)
            @synchronize()

            # recompute softmax dims=2
            in_ms_seq_bounds = in_seq_bounds || tidx + lo_inner ≤ size(ms, 1)
            m_i = in_ms_seq_bounds ? ms[tidx + lo_inner, gidx[1], gidx[2]] : typemax(T)
            for i in 1:gsz
                s_shm[tidx, i] = exp(s_shm[tidx, i] - m_i)
            end

            # compute dv += Δ (emb_dim, L_q) * s_shm (L_q, L_k) = (emb_dim, L_k)
            in_dv_seq_bounds = in_seq_bounds || tidx + lo ≤ size(dv, 2)
            sh_load_emb!(d_shm, dv, lo, in_dv_seq_bounds, Val{false}())
            @synchronize()
            mma!(d_shm, Δ_shm, s_shm, cfg_dv, tidx, mma_acc_fn)
            @synchronize()
            if in_dv_seq_bounds
                for i in 1:emb_dim
                    dv[i, tidx + lo, gidx[1], gidx[2]] = d_shm[i, tidx]
                end
            end

            # compute:
            # - d = Δ' * v: (L_q, emb_dim) * (emb_dim, L_k) = (L_q, L_k)
            # - ((0 - d_i) .+ d) .* s_shm
            sh_load_emb!(d_shm, v, lo, in_dv_seq_bounds, Val{false}())
            @synchronize()

            # TODO prefetch `δ`
            mma!(
                s_shm, Δ_shm, d_shm, cfg_ds, tidx,
                (res, out, x, y) -> begin
                    d_i = in_seq_bounds || x + lo_inner ≤ size(δ, 1) ?
                        @inbounds(δ[x + lo_inner, gidx[1], gidx[2]]) :
                        zero(T)
                    @inbounds out[x, y] * (res - d_i) * scale
                end,
            )

            # dk = s_shm' * q_shm: (L_k, L_q) * (L_q, emb_dim) -> (L_k, emb_dim) -> transpose into shmem
            sh_load_emb!(d_shm, dk, lo, in_k_seq_bounds, Val{false}())
            @synchronize()
            mma!(d_shm, s_shm, q_shm, cfg_dk, tidx, mma_acc_fn)
            @synchronize()
            if in_k_seq_bounds
                for i in 1:emb_dim
                    dk[i, tidx + lo, gidx[1], gidx[2]] = d_shm[i, tidx]
                end
            end

            # compute dq = dot(ds, k) (L_q, L_k) * (L_k, emb_dim)
            in_dq_seq_bounds = in_seq_bounds || tidx + lo_inner ≤ size(dq, 2)
            sh_load_emb!(d_shm, dq, lo_inner, in_dq_seq_bounds, Val{false}())
            @synchronize()
            mma!(d_shm, s_shm, k_shm, cfg_dq, tidx, mma_acc_fn)
            @synchronize()
            if in_dq_seq_bounds
                for i in 1:emb_dim
                    dq[i, tidx + lo_inner, gidx[1], gidx[2]] = d_shm[i, tidx]
                end
            end

            q_offset += gsz
        end
    end
end

@kernel unsafe_indices=true cpu=false inbounds=true function _flash_attention_bwd_preprocess!(
    # Output.
    Δ_scaled::AbstractArray{T, 4},
    δ::AbstractArray{T, 3},
    # Input.
    Δ::AbstractArray{T, 4},
    o::AbstractArray{T, 4},
    ls::AbstractArray{T, 3},
    ::Val{emb_dim}, ::Val{in_seq_bounds},
) where {T, emb_dim, in_seq_bounds}
    gsz = @groupsize()[1]

    tidx = @index(Local)
    gidx = @index(Group, NTuple)
    q_offset = (gidx[1] - 1) * gsz

    in_q_seq_bounds = in_seq_bounds || tidx + q_offset ≤ size(ls, 1)
    in_q_seq_bounds || return

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
    scale = T(inv(sqrt(emb_dim)))

    KL = size(k, 2)
    @assert size(k) == size(v)
    gsz = flash_attention_groupsize(T; emb_dim, target_shmem=64 * 1024) # TODO query available shmem
    @assert gsz ≤ QL

    q_seq_tiles = cld(QL, gsz)
    kv_seq_tiles = cld(KL, gsz)
    threads = (gsz, 1, 1)
    ndrange = (gsz * q_seq_tiles, H, N)

    in_seq_bounds = QL % gsz == 0 && KL % gsz == 0

    kab = get_backend(q)
    Δ_scaled = similar(Δ)
    δ = similar(ls)
    _flash_attention_bwd_preprocess!(kab, threads)(
        # Output.
        Δ_scaled, δ,
        # Input.
        Δ, o, ls,
        Val(emb_dim), Val(in_seq_bounds); ndrange)

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
        o, ms,
        q, k, v, scale,
        Val(emb_dim), Val(q_seq_tiles), Val(kv_seq_tiles), Val(in_seq_bounds); ndrange)

    return dq, dk, dv
end
