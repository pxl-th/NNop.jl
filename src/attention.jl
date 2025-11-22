@kernel unsafe_indices=true cpu=false inbounds=true function _flash_attention_fwd!(
    cfg, cfg_out,
    # outputs
    o::AbstractArray{T, 4}, ms::AbstractArray{T, 3}, ls::AbstractArray{T, 3},
    # inputs
    q::AbstractArray{T, 4}, k::AbstractArray{T, 4}, v::AbstractArray{T, 4},
    scale::T,
    pair::Maybe{AbstractArray{T, 4}},
    q_doc_ids::Maybe{AbstractMatrix{Int32}},
    k_doc_ids::Maybe{AbstractMatrix{Int32}},
    k_tile_start_doc::Maybe{AbstractMatrix{Int32}},
    k_tile_end_doc::Maybe{AbstractMatrix{Int32}},
    ::Val{emb_dim}, ::Val{in_seq_bounds}, ::Val{causal}, ::Val{num_q_per_kv},
) where {T, emb_dim, in_seq_bounds, causal, num_q_per_kv}
    gsz = @groupsize()[1]
    kv_seq_tiles = cld(size(k, 2), gsz)

    # shared-memory buffers ------------------------------------------------
    q_shm = @localmem T (gsz, emb_dim)
    k_shm = @localmem T (emb_dim, gsz)
    s_shm = @localmem T (gsz, gsz)
    o_shm = @localmem T (emb_dim, gsz)
    q_doc_shm = @localmem Int32 (gsz,)
    k_doc_shm = @localmem Int32 (gsz,)
    q_doc_range_shm = @localmem Int32 (2,)
    k_doc_range_shm = @localmem Int32 (2,)

    tidx = @index(Local)
    gidx = @index(Group, NTuple)
    q_offset = (gidx[1] - 1) * gsz
    in_q_seq_bounds = in_seq_bounds || q_offset + tidx ≤ size(q, 2)
    q_head_idx = gidx[2]
    kv_head_idx = cld(q_head_idx, num_q_per_kv)
    doc_mode = !isnothing(q_doc_ids) && !isnothing(k_doc_ids)

    @inline function sh_load_emb!(dest, src, offset, mask::Bool, ::Val{tr}, head_idx) where tr
        @unroll for i in 1:emb_dim
            x, y = tr ? (tidx, i) : (i, tidx)
            @inbounds dest[x, y] = mask ? src[i, tidx + offset, head_idx, gidx[3]] : zero(T)
        end
    end

    @inline function doc_tile_range!(doc_range_shm, doc_shm)
        if tidx == 1
            dmin = typemax(Int32)
            dmax = typemin(Int32)
            @inbounds @unroll for i in 1:gsz
                doc = doc_shm[i]
                doc == 0 && continue
                dmin = min(dmin, doc)
                dmax = max(dmax, doc)
            end
            doc_range_shm[1] = dmin
            doc_range_shm[2] = dmax
        end
    end

    @inline function apply_doc_mask!(s_shm, q_doc_shm, k_doc_shm, k_offset)
        @unroll for i in 1:gsz
            (in_seq_bounds || k_offset + i ≤ size(k, 2)) || break
            same_doc = q_doc_shm[tidx] != 0 && q_doc_shm[tidx] == k_doc_shm[i]
            s_shm[tidx, i] = same_doc ? s_shm[tidx, i] : typemin(T)
        end
    end

    # Load `q` --------------------------------------------------------------
    sh_load_emb!(q_shm, q, q_offset, in_q_seq_bounds, Val{true}(), q_head_idx)
    if doc_mode
        q_pos = q_offset + tidx
        in_q_doc_bounds = in_seq_bounds || q_pos ≤ size(q_doc_ids, 1)
        q_doc_shm[tidx] = in_q_doc_bounds ? q_doc_ids[q_pos, gidx[3]] : Int32(0)
    end
    @unroll for i in 1:emb_dim
        o_shm[i, tidx] = zero(T)
    end
    @synchronize()

    if doc_mode
        doc_tile_range!(q_doc_range_shm, q_doc_shm)
        @synchronize()
    end

    l_i = zero(T)
    m_i = typemin(T)
    end_iter = causal ? gidx[1] : kv_seq_tiles

    k_tile_start = 1
    k_tile_end = end_iter
    if doc_mode && !isnothing(k_tile_start_doc) && !isnothing(k_tile_end_doc)
        dq_min = q_doc_range_shm[1]
        dq_max = q_doc_range_shm[2]
        has_q_docs = dq_min ≤ dq_max
        if has_q_docs
            ndocs_k = size(k_tile_start_doc, 1)
            s = end_iter
            e = 1
            @inbounds for d in dq_min:min(dq_max, ndocs_k)
                ts = k_tile_start_doc[d, gidx[3]]
                te = k_tile_end_doc[d, gidx[3]]
                (ts ≤ te) || continue
                s = min(s, ts)
                e = max(e, te)
            end
            if s ≤ e
                k_tile_start = s
                k_tile_end = min(e, end_iter)
            else
                k_tile_start = 1
                k_tile_end = 0
            end
        else
            k_tile_start = 1
            k_tile_end = 0
        end
    end

    for tile_idx in k_tile_start:k_tile_end
        k_offset = (tile_idx - 1) * gsz
        in_k_seq_bounds = in_seq_bounds || k_offset + tidx ≤ size(k, 2)
        sh_load_emb!(k_shm, k, k_offset, in_k_seq_bounds, Val{false}(), kv_head_idx)
        if doc_mode
            k_pos = k_offset + tidx
            in_k_doc_bounds = in_seq_bounds || k_pos ≤ size(k_doc_ids, 1)
            k_doc_shm[tidx] = in_k_doc_bounds ? k_doc_ids[k_pos, gidx[3]] : Int32(0)
        end
        @synchronize()

        if doc_mode
            doc_tile_range!(k_doc_range_shm, k_doc_shm)
            @synchronize()

            dq_min = q_doc_range_shm[1]
            dq_max = q_doc_range_shm[2]
            dk_min = k_doc_range_shm[1]
            dk_max = k_doc_range_shm[2]
            has_q_docs = dq_min ≤ dq_max
            has_k_docs = dk_min ≤ dk_max
            if has_q_docs && has_k_docs
                overlap_min = max(dq_min, dk_min)
                overlap_max = min(dq_max, dk_max)
                if overlap_min > overlap_max
                    continue
                end
            end
        end

        # ---- scaled Q · Kᵀ ------------------------------------------------
        mma!(s_shm, q_shm, k_shm, cfg, tidx, (res, c_shm, x, y) -> res * scale)
        @synchronize()

        # ---- add pair features -------------------------------------------
        if !isnothing(pair)
            @unroll for i in 1:gsz
                (in_seq_bounds || k_offset + i ≤ size(k, 2)) || break
                in_q_seq_bounds || break
                s_shm[tidx, i] += pair[q_head_idx, q_offset + tidx, k_offset + i, gidx[3]]
            end
        end

        # ---- causal / pad masking ----------------------------------------
        if causal
            @unroll for i in 1:gsz
                (in_seq_bounds || k_offset + i ≤ size(k, 2)) || break
                s_shm[tidx, i] = (tidx + q_offset ≥ i + k_offset) ? s_shm[tidx, i] : typemin(T)
            end
        end
        if doc_mode
            apply_doc_mask!(s_shm, q_doc_shm, k_doc_shm, k_offset)
        end

        # ---- online soft-max ---------------------------------------------
        m_ij = typemin(T)
        @unroll for i in 1:gsz
            (in_seq_bounds || k_offset + i ≤ size(k, 2)) || break
            m_ij = max(m_ij, s_shm[tidx, i])
        end

        has_valid = m_ij != typemin(T)

        l_ij = zero(T)
        if has_valid
            @unroll for i in 1:gsz
                (in_seq_bounds || k_offset + i ≤ size(k, 2)) || break
                tmp = exp(s_shm[tidx, i] - m_ij)
                l_ij += tmp
                s_shm[tidx, i] = tmp
            end
        else
            @unroll for i in 1:gsz
                s_shm[tidx, i] = zero(T)
            end
        end
        @synchronize()

        if has_valid
            m_i_new = max(m_i, m_ij)
            α = exp(m_i - m_i_new)
            β = exp(m_ij - m_i_new)
            l_i_new = α * l_i + β * l_ij
        else
            m_i_new = m_i
            α = one(T)
            β = zero(T)
            l_i_new = l_i
        end

        p_scale = zero(T)
        o_scale = one(T)
        if l_i_new != zero(T)
            p_scale = β / l_i_new
            o_scale = l_i / l_i_new * α
        end

        @unroll for i in 1:gsz
            s_shm[tidx, i] *= p_scale
        end
        @unroll for i in 1:emb_dim
            o_shm[i, tidx] *= o_scale
        end

        # ---- P · V --------------------------------------------------------
        sh_load_emb!(k_shm, v, k_offset, in_k_seq_bounds, Val{false}(), kv_head_idx)
        @synchronize()
        mma!(o_shm, s_shm, k_shm, cfg_out, tidx, mma_acc_fn)
        @synchronize()

        m_i = m_i_new
        l_i = l_i_new
        k_offset += gsz
    end

    # ---- write-back -------------------------------------------------------
    if in_seq_bounds || in_q_seq_bounds
        @unroll for i in 1:emb_dim
            o[i, tidx + q_offset, q_head_idx, gidx[3]] = o_shm[i, tidx]
        end
        ms[tidx + q_offset, q_head_idx, gidx[3]] = m_i
        ls[tidx + q_offset, q_head_idx, gidx[3]] = l_i
    end
end



function _flash_attention(
    q::AbstractArray{T,4}, k::AbstractArray{T,4}, v::AbstractArray{T,4},
    pair::Union{Nothing,AbstractArray{T,4}} = nothing;
    causal::Bool,
    q_lengths=nothing,
    k_lengths=nothing,
) where T
    emb_dim, QL, QH, B = size(q)
    KL, KVH = size(k, 2), size(k, 3)
    @assert size(k) == size(v)
    @assert size(k, 1) == emb_dim
    @assert size(k, 4) == B
    ispow2(emb_dim) || error("Only power-of-2 embedding dims are supported.")

    @assert QH % KVH == 0 "Number of query heads ($QH) must be divisible by number of KV heads ($KVH)"
    num_q_per_kv = QH ÷ KVH

    kab          = get_backend(q)
    target_shmem = shared_memory(kab, KA.device(kab))
    gsz          = flash_attention_groupsize(T; emb_dim, target_shmem)

    q_seq_tiles, kv_seq_tiles = cld.((QL, KL), gsz)
    threads   = (gsz, 1, 1)
    ndrange   = (gsz * q_seq_tiles, QH, B)
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
    ms = KA.allocate(kab, eltype(o), (QL,QH,B))
    ls = KA.allocate(kab, eltype(o), (QL,QH,B))

    q_doc_ids = nothing
    k_doc_ids = nothing
    k_tile_start_doc = nothing
    k_tile_end_doc = nothing
    if q_lengths !== nothing || k_lengths !== nothing
        (q_lengths === nothing || k_lengths === nothing) &&
            error("Both q_lengths and k_lengths must be provided together.")
        @assert size(q_lengths, 2) == B
        @assert size(k_lengths, 2) == B
        q_doc_ids = build_doc_ids(kab, q_lengths, QL, B)
        k_doc_ids = build_doc_ids(kab, k_lengths, KL, B)
        k_tile_start_doc, k_tile_end_doc = build_k_tile_ranges(kab, k_lengths, gsz, KL, B)
        if causal
            size(q_lengths) == size(k_lengths) ||
                error("For causal attention, q_lengths and k_lengths must match shapes.")
            all(q_lengths .== k_lengths) ||
                error("For causal attention, q_lengths and k_lengths must be equal.")
        end
    end

    _flash_attention_fwd!(kab, threads)(
        cfg, cfg_out,
        o, ms, ls, q, k, v, scale, pair,
        q_doc_ids, k_doc_ids, k_tile_start_doc, k_tile_end_doc,
        Val(emb_dim), Val(in_bounds), Val(causal), Val(num_q_per_kv);   # flags
        ndrange)

    return o, ms, ls
end

function flash_attention_shmem_fwd(::Type{T}; emb_dim::Int, groupsize::Int)::Int where T
    doc_bytes = sizeof(Int32) * (2 * groupsize + 4)
    return sizeof(T) * (
        3 * groupsize * emb_dim + # q_shm, k_shm, o_shm
        groupsize * groupsize     # s_shm
    ) + doc_bytes
end

function flash_attention_shmem_bwd(::Type{T};
    emb_dim::Int, groupsize::Int, qk_fp16::Bool,
)::Int where T
    doc_bytes = sizeof(Int32) * (2 * groupsize + 4)
    return sizeof(T) * (2 * groupsize * emb_dim + groupsize * groupsize) +
        sizeof(qk_fp16 ? Float16 : Float32) * 2 * groupsize * emb_dim +
        doc_bytes
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

@kernel function _build_doc_ids!(
    doc_ids::AbstractMatrix{Int32},
    lengths,
    ndocs::Int32,
)
    idx = @index(Global, Linear)
    L = size(doc_ids, 1)
    B = size(doc_ids, 2)
    total = L * B
    if idx <= total

        pos = (idx - 1) % L + 1
        b   = (idx - 1) ÷ L + 1

        remaining = pos
        doc::Int32 = 0
        @inbounds for d in 1:ndocs
            len = Int32(lengths[d, b])
            len == 0 && continue
            if remaining <= len
                doc = Int32(d)
                break
            end
            remaining -= len
        end
        doc_ids[pos, b] = doc
    end
end

function build_doc_ids(kab, lengths, L::Int, B::Int)
    ndocs = size(lengths, 1)
    @assert size(lengths, 2) == B
    doc_ids = KA.allocate(kab, Int32, (L, B))

    threads = 256
    ndrange = L * B
    _build_doc_ids!(kab, threads)(doc_ids, lengths, Int32(ndocs); ndrange)

    return doc_ids
end

@kernel function _build_k_tile_ranges!(
    tile_start::AbstractMatrix{Int32},
    tile_end::AbstractMatrix{Int32},
    lengths,
    gsz::Int32,
)
    idx = @index(Global, Linear)
    ndocs = size(lengths, 1)
    B = size(lengths, 2)
    total = ndocs * B
    if idx <= total
        d = (idx - 1) % ndocs + 1
        b = (idx - 1) ÷ ndocs + 1

        pos = 1
        @inbounds for i in 1:(d-1)
            pos += Int(lengths[i, b])
        end
        len_d = Int(lengths[d, b])
        if len_d == 0
            tile_start[d, b] = Int32(1)
            tile_end[d, b] = Int32(0)
        else
            start_pos = pos
            end_pos = pos + len_d - 1
            ts = (start_pos - 1) ÷ gsz + 1
            te = (end_pos   - 1) ÷ gsz + 1
            tile_start[d, b] = Int32(ts)
            tile_end[d, b] = Int32(te)
        end
    end
end

function build_k_tile_ranges(kab, lengths, gsz::Int, L::Int, B::Int)
    ndocs = size(lengths, 1)
    @assert size(lengths, 2) == B
    tile_start = KA.allocate(kab, Int32, (ndocs, B))
    tile_end   = KA.allocate(kab, Int32, (ndocs, B))

    threads = 64
    ndrange = ndocs * B
    _build_k_tile_ranges!(kab, threads)(tile_start, tile_end, lengths, Int32(gsz); ndrange)

    return tile_start, tile_end
end
