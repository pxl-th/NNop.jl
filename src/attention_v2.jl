struct FATileConfig{BM, BK, BN, TM, TN, rev_a, rev_b, rev_c} end

@kernel cpu=false inbounds=true function _flash_attention_fwd_v2!(
    cfg, cfg_out,
    # Output.
    o::AbstractArray{T, 4}, ms::AbstractArray{T, 3}, ls::AbstractArray{T, 3},
    # Input.
    q::AbstractArray{T, 4}, k::AbstractArray{T, 4}, v::AbstractArray{T, 4},
    ::Val{emb_dim}, ::Val{n_seq_tiles},
) where {T, emb_dim, n_seq_tiles}
    gsz = prod(@groupsize())

    # `v` shares shmem with `k`.
    q_shm = @localmem T (gsz, emb_dim)
    k_shm = @localmem T (emb_dim, gsz)
    s_shm = @localmem T (gsz, gsz)
    o_shm = @localmem T (emb_dim, gsz)

    tidx = @index(Local)
    gidx = @index(Group, NTuple)
    q_offset = (gidx[1] - 1) * gsz

    # Load transposed `q` and `k` into shmem (done only once per workgroup).
    # loop idx accross emb dim, thread idx accross L dim
    for i in 1:emb_dim
        q_shm[tidx, i] = q[i, tidx + q_offset, gidx[2], gidx[3]]
        o_shm[i, tidx] = zero(T)
    end
    @synchronize()

    l_i = zero(T)
    m_i = typemin(T)
    k_offset = 0
    # for _ in 1:gidx[1] # TODO use when causal
    for _ in 1:n_seq_tiles
        for i in 1:emb_dim
            k_shm[i, tidx] = k[i, tidx + k_offset, gidx[2], gidx[3]]
        end
        @synchronize()

        # compute q' * k (L_q, L_k)
        mma!(s_shm, q_shm, k_shm, cfg, tidx, false)
        @synchronize()

        # find max(qk; dims=2)
        m_ij = typemin(T)
        for i in 1:gsz
            m_ij = max(m_ij, s_shm[tidx, i])
        end

        # compute softmax dims=2
        l_ij = zero(T)
        for i in 1:gsz
            tmp = exp(s_shm[tidx, i] - m_ij)
            l_ij += tmp
            s_shm[tidx, i] = tmp
        end
        @synchronize()

        m_i_new = max(m_i, m_ij)
        α = exp(m_i - m_i_new)
        β = exp(m_ij - m_i_new)
        l_i_new = α * l_i + β * l_ij

        p_scale = β / l_i_new
        o_scale = l_i / l_i_new * α

        for i in 1:gsz
            s_shm[tidx, i] *= p_scale
        end
        for i in 1:emb_dim
            o_shm[i, tidx] *= o_scale
        end
        # load `v` into `k_shm` (shared shmem).
        for i in 1:emb_dim
            k_shm[i, tidx] = v[i, tidx + k_offset, gidx[2], gidx[3]]
        end
        @synchronize()

        # (q' * k) * v' (L_q, emb_dim)
        mma!(o_shm, s_shm, k_shm, cfg_out, tidx, true)
        @synchronize()

        m_i = m_i_new
        l_i = l_i_new
        k_offset += gsz
    end

    for i in 1:emb_dim
        o[i, tidx + q_offset, gidx[2], gidx[3]] = o_shm[i, tidx]
    end

    # Store for the backward pass.
    ms[tidx + q_offset, gidx[2], gidx[3]] = m_i
    ls[tidx + q_offset, gidx[2], gidx[3]] = l_i
end

Base.@propagate_inbounds function mma!(
    c_shm::AbstractMatrix{T},
    a_shm::AbstractMatrix,
    b_shm::AbstractMatrix,
    cfg::Type{FATileConfig{BM, BK, BN, TM, TN, rev_a, rev_b, rev_c}},
    tidx,
    accumulate::Bool,
) where {T, BM, BK, BN, TM, TN, rev_a, rev_b, rev_c}
    thread_row = (tidx - 1) ÷ (BN ÷ TN)
    thread_col = (tidx - 1) % (BN ÷ TN)
    row_offset = thread_row * TM
    col_offset = thread_col * TN

    results = zeros(MMatrix{TM, TN, T})
    reg_m = MVector{TM, T}(undef)
    reg_n = MVector{TN, T}(undef)

    for dot_idx in 1:BK
        @unroll for reg_idx in 1:TM
            x, y = rev_a ? (dot_idx, row_offset + reg_idx) : (row_offset + reg_idx, dot_idx)
            reg_m[reg_idx] = a_shm[x, y]
        end
        @unroll for reg_idx in 1:TN
            x, y = rev_b ? (col_offset + reg_idx, dot_idx) : (dot_idx, col_offset + reg_idx)
            reg_n[reg_idx] = b_shm[x, y]
        end
        @unroll for res_idx_m in 1:TM
            @unroll for res_idx_n in 1:TN
                # TODO use fma
                results[res_idx_m, res_idx_n] += reg_m[res_idx_m] * reg_n[res_idx_n]
            end
        end
    end

    @unroll for res_idx_m in 1:TM
        @unroll for res_idx_n in 1:TN
            x, y = rev_c ?
                (col_offset + res_idx_n, row_offset + res_idx_m) :
                (row_offset + res_idx_m, col_offset + res_idx_n)
            c_shm[x, y] = accumulate ?
                # TODO use fma
                (c_shm[x, y] + results[res_idx_m, res_idx_n]) :
                results[res_idx_m, res_idx_n]
        end
    end
    return
end

Base.@propagate_inbounds function mma!(
    c_shm::AbstractMatrix{T},
    a_shm::AbstractMatrix,
    b_shm::AbstractMatrix,
    cfg::Type{FATileConfig{BM, BK, BN, TM, TN, rev_a, rev_b, rev_c}},
    tidx,
    fn
) where {T, BM, BK, BN, TM, TN, rev_a, rev_b, rev_c}
    thread_row = (tidx - 1) ÷ (BN ÷ TN)
    thread_col = (tidx - 1) % (BN ÷ TN)
    row_offset = thread_row * TM
    col_offset = thread_col * TN

    results = zeros(MMatrix{TM, TN, T})
    reg_m = MVector{TM, T}(undef)
    reg_n = MVector{TN, T}(undef)

    for dot_idx in 1:BK
        @unroll for reg_idx in 1:TM
            x, y = rev_a ? (dot_idx, row_offset + reg_idx) : (row_offset + reg_idx, dot_idx)
            reg_m[reg_idx] = a_shm[x, y]
        end
        @unroll for reg_idx in 1:TN
            x, y = rev_b ? (col_offset + reg_idx, dot_idx) : (dot_idx, col_offset + reg_idx)
            reg_n[reg_idx] = b_shm[x, y]
        end
        @unroll for res_idx_m in 1:TM
            @unroll for res_idx_n in 1:TN
                # TODO use fma
                results[res_idx_m, res_idx_n] += reg_m[res_idx_m] * reg_n[res_idx_n]
            end
        end
    end

    @unroll for res_idx_m in 1:TM
        @unroll for res_idx_n in 1:TN
            x, y = rev_c ?
                (col_offset + res_idx_n, row_offset + res_idx_m) :
                (row_offset + res_idx_m, col_offset + res_idx_n)
            c_shm[x, y] = fn(c_shm[x, y], results[res_idx_m, res_idx_n], x, y)
        end
    end
    return
end

function flash_attention_v2(
    q::AbstractArray{T, 4}, k::AbstractArray{T, 4}, v::AbstractArray{T, 4},
) where T
    gsz = 64
    emb_dim, L, H, N = size(q)

    @assert L ≥ gsz
    n_seq_tiles = cld(L, gsz)
    threads = (gsz, 1, 1)
    ndrange = (gsz * n_seq_tiles, H, N)

    kab = get_backend(q)
    o = similar(q)
    ms = KA.allocate(kab, eltype(o), (L, H, N))
    ls = KA.allocate(kab, eltype(o), (L, H, N))

    # In mma, each thread processes `TM × TN` output values
    # so `TM × TN × gsz` covers the whole output tile.
    #
    # mma config for Q' * K tile: (L_q, emb_dim) * (emb_dim, L_k).
    BM, BK, BN = gsz, emb_dim, gsz
    TM, TN = 8, 8
    cfg = FATileConfig{BM, BK, BN, TM, TN, false, false, false}
    @assert prod(threads) == (BM * BN) ÷ (TM * TN)

    # mma config for (Q' * K) * V' (L_q, L_k) * (L_k, emb_dim)
    BM, BK, BN = gsz, gsz, emb_dim
    TM, TN = 8, 8
    cfg_out = FATileConfig{BM, BK, BN, TM, TN, false, true, true}
    @assert prod(threads) == (BM * BN) ÷ (TM * TN)

    _flash_attention_fwd_v2!(kab, threads)(
        cfg, cfg_out,
        o, ms, ls,
        q, k, v,
        Val(emb_dim), Val(n_seq_tiles);
        ndrange)

    return o, ms, ls
end
