@kernel unsafe_indices=true cpu=false inbounds=true function _flash_attention_fwd!(
    cfg, cfg_out,
    # Output.
    o::AbstractArray{T, 4}, ms::AbstractArray{T, 3}, ls::AbstractArray{T, 3},
    # Input.
    q::AbstractArray{T, 4}, k::AbstractArray{T, 4}, v::AbstractArray{T, 4},
    ::Val{emb_dim}, ::Val{kv_seq_tiles}, ::Val{in_seq_bounds},
) where {T, emb_dim, kv_seq_tiles, in_seq_bounds}
    gsz = @groupsize()[1]

    # `v` shares shmem with `k`.
    q_shm = @localmem T (gsz, emb_dim)
    k_shm = @localmem T (emb_dim, gsz)
    s_shm = @localmem T (gsz, gsz)
    o_shm = @localmem T (emb_dim, gsz)

    tidx = @index(Local)
    gidx = @index(Group, NTuple)
    q_offset = (gidx[1] - 1) * gsz
    in_q_seq_bounds = in_seq_bounds || q_offset + tidx ≤ size(q, 2)

    @inline function sh_load_emb!(dest, source, offset, mask::Bool, ::Val{transposed}) where transposed
        @unroll for i in 1:emb_dim
            x, y = transposed ? (tidx, i) : (i, tidx)
            @inbounds dest[x, y] = mask ? source[i, tidx + offset, gidx[2], gidx[3]] : zero(T)
        end
    end

    # Load transposed `q` and `k` into shmem (done only once per workgroup).
    # loop idx accross emb dim, thread idx accross L dim
    sh_load_emb!(q_shm, q, q_offset, in_q_seq_bounds, Val{true}())
    for i in 1:emb_dim
        o_shm[i, tidx] = zero(T)
    end
    @synchronize()

    l_i = zero(T)
    m_i = typemin(T)
    k_offset = 0
    # for _ in 1:gidx[1] # TODO use when causal
    for _ in 1:kv_seq_tiles
        in_k_seq_bounds = in_seq_bounds || k_offset + tidx ≤ size(k, 2)
        sh_load_emb!(k_shm, k, k_offset, in_k_seq_bounds, Val{false}())
        @synchronize()

        # compute q' * k (L_q, L_k)
        mma!(s_shm, q_shm, k_shm, cfg, tidx, mma_non_acc_fn)
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
        sh_load_emb!(k_shm, v, k_offset, in_k_seq_bounds, Val{false}())
        @synchronize()
        # (q' * k) * v' (L_q, emb_dim)
        mma!(o_shm, s_shm, k_shm, cfg_out, tidx, mma_acc_fn)
        @synchronize()

        m_i = m_i_new
        l_i = l_i_new
        k_offset += gsz
    end

    if in_seq_bounds || in_q_seq_bounds
        for i in 1:emb_dim
            o[i, tidx + q_offset, gidx[2], gidx[3]] = o_shm[i, tidx]
        end
        # Store for the backward pass.
        ms[tidx + q_offset, gidx[2], gidx[3]] = m_i
        ls[tidx + q_offset, gidx[2], gidx[3]] = l_i
    end
end

function flash_attention(
    q::AbstractArray{T, 4}, k::AbstractArray{T, 4}, v::AbstractArray{T, 4},
) where T
    emb_dim, QL, H, N = size(q)
    KL = size(k, 2)
    @assert size(k) == size(v)
    # TODO LRU cache
    gsz = flash_attention_groupsize(T; emb_dim, target_shmem=64 * 1024) # TODO query available shmem

    q_seq_tiles = cld(QL, gsz)
    kv_seq_tiles = cld(KL, gsz)
    threads = (gsz, 1, 1)
    ndrange = (gsz * q_seq_tiles, H, N)

    kab = get_backend(q)
    o = similar(q)
    ms = KA.allocate(kab, eltype(o), (QL, H, N))
    ls = KA.allocate(kab, eltype(o), (QL, H, N))

    # In mma, each thread processes `TM × TN` output values
    # so `TM × TN × gsz` covers the whole output tile.
    #
    # mma config for Q' * K tile: (L_q, emb_dim) * (emb_dim, L_k).
    BM, BK, BN = gsz, emb_dim, gsz
    TM, TN = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg = FATileConfig{BM, BK, BN, TM, TN, false, false, false}

    # mma config for (Q' * K) * V' (L_q, L_k) * (L_k, emb_dim)
    BM, BK, BN = gsz, gsz, emb_dim
    TM, TN = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg_out = FATileConfig{BM, BK, BN, TM, TN, false, true, true}

    in_seq_bounds = QL % gsz == 0 && KL % gsz == 0
    _flash_attention_fwd!(kab, threads)(
        cfg, cfg_out,
        o, ms, ls,
        q, k, v,
        Val(emb_dim), Val(kv_seq_tiles), Val(in_seq_bounds);
        ndrange)
    return o, ms, ls
end

function flash_attention_shmem_fwd(::Type{T}; emb_dim::Int, groupsize::Int)::Int where T
    return sizeof(T) * (
        3 * groupsize * emb_dim + # q_shm, k_shm, o_shm
        groupsize * groupsize     # s_shm
    )
end

function flash_attention_shmem_bwd(::Type{T};
    emb_dim::Int, groupsize::Int, qk_fp16::Bool,
)::Int where T
    return sizeof(T) * (2 * groupsize * emb_dim + groupsize * groupsize) +
        sizeof(qk_fp16 ? Float16 : Float32) * 2 * groupsize * emb_dim
end

function flash_attention_groupsize(::Type{T}; emb_dim::Int, target_shmem::Int) where T
    # TODO
    # - return `qk_fp16` to configure kernel
    # - optional qk_fp16
    # qk_fp16s = (false, true)
    qk_fp16s = (true,)
    for qk_fp16 in qk_fp16s, groupsize in (256, 128, 64, 32)
        shmem = flash_attention_shmem_bwd(T; emb_dim, groupsize, qk_fp16)
        shmem ≤ target_shmem && return groupsize
    end
    error("Failed to find groupsize for Flash Attention that satisfies Shared Memory constraint.")
end

function flash_attention_mma_thread_cfg(groupsize::Int; BM::Int, BN::Int)::Tuple{Int, Int}
    tmp = (BM * BN) ÷ groupsize
    x = Int(log2(tmp))
    TM, TN = if iseven(x)
        2^(x / 2), 2^(x / 2)
    else
        2^((x + 1) / 2), 2^((x - 1) / 2)
    end

    @assert groupsize == (BM * BN) ÷ (TM * TN)
    return TM, TN
end
