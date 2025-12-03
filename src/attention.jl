@kernel unsafe_indices=true cpu=false inbounds=true function _flash_attention_fwd!(
    cfg, cfg_out,
    # outputs
    o::AbstractArray{T, 4}, ms::AbstractArray{T, 3}, ls::AbstractArray{T, 3},
    # inputs
    q::AbstractArray{T, 4}, k::AbstractArray{T, 4}, v::AbstractArray{T, 4},
    scale::T,
    pair::Maybe{AbstractArray{T, 4}},
    kpad_mask::Maybe{AbstractMatrix{Bool}},
    ::Val{emb_dim}, ::Val{in_seq_bounds}, ::Val{causal},
) where {T, emb_dim, in_seq_bounds, causal}
    gsz = @groupsize()[1]
    kv_seq_tiles = cld(size(k, 2), gsz)

    # shared-memory buffers ------------------------------------------------
    q_shm = @localmem T (gsz, emb_dim)
    k_shm = @localmem T (emb_dim, gsz)
    s_shm = @localmem T (gsz, gsz)
    o_shm = @localmem T (emb_dim, gsz)

    tidx = @index(Local)
    gidx = @index(Group, NTuple)
    q_offset = (gidx[1] - 1) * gsz
    in_q = in_seq_bounds || q_offset + tidx ≤ size(q, 2)

    @inline function sh_load_emb!(dest, src, offset, mask::Bool, ::Val{tr}) where tr
        @unroll for i in 1:emb_dim
            x, y = tr ? (tidx, i) : (i, tidx)
            @inbounds dest[x, y] = mask ? src[i, tidx + offset, gidx[2], gidx[3]] : zero(T)
        end
    end

    # Load `q` --------------------------------------------------------------
    sh_load_emb!(q_shm, q, q_offset, in_q, Val{true}())
    @unroll for i in 1:emb_dim
        o_shm[i, tidx] = zero(T)
    end
    @synchronize()

    l_i = zero(T)
    m_i = typemin(T)
    k_offset = 0
    end_iter = causal ? gidx[1] : kv_seq_tiles

    for _ in 1:end_iter
        in_k = in_seq_bounds || k_offset + tidx ≤ size(k, 2)
        sh_load_emb!(k_shm, k, k_offset, in_k, Val{false}())
        @synchronize()

        # ---- scaled Q · Kᵀ ------------------------------------------------
        mma!(s_shm, q_shm, k_shm, cfg, tidx, (res, c_shm, x, y) -> res * scale)
        @synchronize()

        # ---- add pair features -------------------------------------------
        if !isnothing(pair)
            @unroll for i in 1:gsz
                in_seq_bounds || (in_q && k_offset + i ≤ size(k, 2)) || break
                s_shm[tidx, i] += pair[gidx[2], q_offset + tidx, k_offset + i, gidx[3]]
            end
        end

        # ---- causal / pad masking ----------------------------------------
        if causal
            @unroll for i in 1:gsz
                (in_seq_bounds || k_offset + i ≤ size(k, 2)) || break
                s_shm[tidx, i] = (tidx + q_offset ≥ i + k_offset) ? s_shm[tidx, i] : typemin(T)
            end
        end
        if !isnothing(kpad_mask)
            @unroll for i in 1:gsz
                (in_seq_bounds || k_offset + i ≤ size(k, 2)) || break
                valid = kpad_mask[k_offset + i, gidx[3]]
                s_shm[tidx, i] = valid ? s_shm[tidx, i] : typemin(T)
            end
        end

        # ---- online soft-max ---------------------------------------------
        m_ij = typemin(T)
        @unroll for i in 1:gsz
            (in_seq_bounds || k_offset + i ≤ size(k, 2)) || break
            m_ij = max(m_ij, s_shm[tidx, i])
        end

        l_ij = zero(T)
        @unroll for i in 1:gsz
            (in_seq_bounds || k_offset + i ≤ size(k, 2)) || break
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

        @unroll for i in 1:gsz
            s_shm[tidx, i] *= p_scale
        end
        @unroll for i in 1:emb_dim
            o_shm[i, tidx] *= o_scale
        end

        # ---- P · V --------------------------------------------------------
        sh_load_emb!(k_shm, v, k_offset, in_k, Val{false}())
        @synchronize()
        mma!(o_shm, s_shm, k_shm, cfg_out, tidx, mma_acc_fn)
        @synchronize()

        m_i = m_i_new
        l_i = l_i_new
        k_offset += gsz
    end

    # ---- write-back -------------------------------------------------------
    if in_seq_bounds || in_q
        @unroll for i in 1:emb_dim
            o[i, tidx + q_offset, gidx[2], gidx[3]] = o_shm[i, tidx]
        end
        ms[tidx + q_offset, gidx[2], gidx[3]] = m_i
        ls[tidx + q_offset, gidx[2], gidx[3]] = l_i
    end
end

function _flash_attention(
    q::AbstractArray{T,4}, k::AbstractArray{T,4}, v::AbstractArray{T,4},
    pair::Union{Nothing,AbstractArray{T,4}} = nothing;
    causal::Bool, kpad_mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
) where T
    emb_dim, QL, H, B = size(q)
    KL = size(k, 2)
    @assert size(k) == size(v)
    ispow2(emb_dim) || error("Only power-of-2 embedding dims are supported.")

    kab          = get_backend(q)
    target_shmem = shared_memory(kab, KA.device(kab))
    gsz          = flash_attention_groupsize(T; emb_dim, target_shmem)

    q_seq_tiles, kv_seq_tiles = cld.((QL, KL), gsz)
    threads   = (gsz, 1, 1)
    ndrange   = (gsz * q_seq_tiles, H, B)
    in_bounds = QL % gsz == 0 && KL % gsz == 0
    scale     = T(inv(sqrt(emb_dim)))

    # mma tile configs ------------------------------------------------------
    BM, BK, BN = gsz, emb_dim, gsz
    TM, TN     = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg        = FATileConfig{BM,BK,BN,TM,TN,false,false,false}

    BM, BK, BN = gsz, gsz, emb_dim
    TM, TN     = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg_out    = FATileConfig{BM,BK,BN,TM,TN,false,true,true}

    # ----------------------------------------------------------------------
    o  = similar(q)
    ms = KA.allocate(kab, eltype(o), (QL,H,B))
    ls = KA.allocate(kab, eltype(o), (QL,H,B))

    _flash_attention_fwd!(kab, threads)(
        cfg, cfg_out,
        o, ms, ls, q, k, v, scale, pair, kpad_mask,
        Val(emb_dim), Val(in_bounds), Val(causal);   # flags
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

function flash_attention_groupsize(::Type{T}; emb_dim::Int, target_shmem::UInt64) where T
    # TODO
    # - return `qk_fp16` to configure kernel
    # - optional qk_fp16
    # qk_fp16s = (false, true)
    # TODO prefer bigger groupsize?
    qk_fp16s = (true,)
    for qk_fp16 in qk_fp16s, groupsize in (256, 128, 64, 32, 16)
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
