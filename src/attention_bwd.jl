@kernel unsafe_indices=true cpu=false inbounds=true function _flash_attention_bwd!(
    cfg, cfg_dv, cfg_dk, cfg_dq, cfg_ds,
    dq::AbstractArray{T,4}, dk::AbstractArray{T,4}, dv::AbstractArray{T,4},
    dpair::AbstractArray{T,4}, # ← grads for pairs
    Δ::AbstractArray{T,4}, δ::AbstractArray{T,3},
    o::AbstractArray{T,4}, ms::AbstractArray{T,3},
    q::AbstractArray{T,4}, k::AbstractArray{T,4}, v::AbstractArray{T,4},
    scale::T,
    pair::Maybe{AbstractArray{T,4}}, # ← pair tensor
    kpad_mask::Maybe{AbstractMatrix{Bool}},
    ::Val{emb_dim}, ::Val{in_seq_bounds}, ::Val{causal}, ::Val{num_q_per_kv},
) where {T, emb_dim, in_seq_bounds, causal, num_q_per_kv}
    gsz = @groupsize()[1]
    q_seq_tiles = cld(size(q, 2), gsz)
    kv_seq_tiles = cld(size(k, 2), gsz)

    # ------------------------------------------------------------------ shmem
    q_shm = @localmem Float16 (gsz, emb_dim)
    k_shm = @localmem Float16 (emb_dim, gsz)
    s_shm = @localmem T       (gsz, gsz)   # scores / dS
    Δ_shm = @localmem T       (emb_dim, gsz)
    d_shm = @localmem T       (emb_dim, gsz)

    tidx = @index(Local)
    gidx = @index(Group, NTuple)          # (head, batch) in this kernel
    
    # Map query head to KV head for grouped-query attention
    q_head_idx = gidx[1]
    kv_head_idx = cld(q_head_idx, num_q_per_kv)

    @inline function sh_load_emb!(dest, src, offset, mask::Bool, ::Val{tr}, head_idx) where tr
        @unroll for i in 1:emb_dim
            x, y = tr ? (tidx, i) : (i, tidx)
            @inbounds dest[x,y] = mask ? src[i, tidx+offset, head_idx, gidx[2]] : zero(T)
        end
    end

    # --------------------------------------------------------------- 2-nested
    for start_n in 1:kv_seq_tiles                     # iterate key-tiles
        lo_k     = (start_n - 1) * gsz                # column offset
        q_offset = causal ? lo_k : 0                  # starting query row

        in_k_ok = in_seq_bounds || tidx + lo_k ≤ size(k,2)
        sh_load_emb!(k_shm, k, lo_k, in_k_ok, Val(false), kv_head_idx)
        @synchronize()

        start_m = causal ? start_n : 1                # iterate query-tiles
        for sm in start_m:q_seq_tiles
            lo_q = (sm - 1) * gsz                    # query offset

            # ------------- load Δ and Q ---------------------------------
            in_q_ok = in_seq_bounds || tidx + q_offset ≤ size(q,2)
            sh_load_emb!(Δ_shm, Δ, q_offset, in_q_ok, Val(false), q_head_idx)
            sh_load_emb!(q_shm, q, q_offset, in_q_ok, Val(true), q_head_idx)
            @synchronize()

            # ------------- recompute raw scores -------------------------
            mma!(s_shm, q_shm, k_shm, cfg, tidx,
                (res,_,__,___) -> res * scale)
            @synchronize()

            # ---- add pair logits so that soft-max matches forward ------
            if !isnothing(pair)
                @unroll for j in 1:gsz
                    (in_seq_bounds || lo_k + j ≤ size(pair,3)) || break
                    (in_seq_bounds || q_offset + tidx ≤ size(pair,2)) || break
                    s_shm[tidx, j] += pair[q_head_idx, q_offset + tidx, lo_k + j, gidx[2]]
                end
            end

            # ---------------- causal / pad masks ------------------------
            if causal
                @unroll for j in 1:gsz
                    (in_seq_bounds || j + lo_k ≤ size(k, 2)) || break
                    s_shm[tidx, j] = tidx + q_offset ≥ j + lo_k ? s_shm[tidx, j] : typemin(T)
                end
            end
            if !isnothing(kpad_mask)
                @unroll for j in 1:gsz
                    (in_seq_bounds || j + lo_k ≤ size(k, 2)) || break
                    valid = kpad_mask[j + lo_k, gidx[2]]
                    s_shm[tidx, j] = valid ? s_shm[tidx, j] : typemin(T)
                end
            end

            # ---------------- soft-max reconstruction -------------------
            in_ms = in_seq_bounds || tidx + lo_q ≤ size(ms,1)
            m_i   = in_ms ? ms[tidx + lo_q, q_head_idx, gidx[2]] : typemax(T)
            @unroll for j in 1:gsz
                s_shm[tidx, j] = exp(s_shm[tidx, j] - m_i)
            end

            # -------------------- dV ------------------------------------
            in_dv = in_seq_bounds || tidx + lo_k ≤ size(dv,2)
            # Don't load existing dv - compute fresh contribution for atomic add
            @unroll for i in 1:emb_dim
                d_shm[i, tidx] = zero(T)
            end
            @synchronize()
            mma!(d_shm, Δ_shm, s_shm, cfg_dv, tidx, mma_acc_fn)
            @synchronize()
            if in_dv
                @unroll for i in 1:emb_dim
                    KA.@atomic dv[i, tidx + lo_k, kv_head_idx, gidx[2]] += d_shm[i, tidx]
                end
            end

            # -------------------- dS (back into s_shm) -------------------
            sh_load_emb!(d_shm, v, lo_k, in_dv, Val(false), kv_head_idx)
            @synchronize()
            # TODO prefetch δ?
            mma!(s_shm, Δ_shm, d_shm, cfg_ds, tidx,
                 (res, out, x, y) -> begin
                     d_i = if in_seq_bounds || x + lo_q ≤ size(δ, 1)
                         @inbounds δ[x + lo_q, q_head_idx, gidx[2]]
                     else
                         zero(T)
                     end
                     out[x,y] * (res - d_i) * scale
                 end)
            @synchronize()

            # -------------------- dpair ----------------------------------
            if !isnothing(pair)
                row = tidx + lo_q
                @unroll for j in 1:gsz
                    col = j + lo_k
                    (in_seq_bounds || col ≤ size(dpair, 3)) || break
                    if (in_seq_bounds || row ≤ size(dpair, 2))
                        dpair[q_head_idx, row, col, gidx[2]] = s_shm[tidx, j] / scale
                    end
                end
            end
            # -------------------- dK ------------------------------------
            # Don't load existing dk - compute fresh contribution for atomic add
            @unroll for i in 1:emb_dim
                d_shm[i, tidx] = zero(T)
            end
            @synchronize()
            mma!(d_shm, s_shm, q_shm, cfg_dk, tidx, mma_acc_fn)
            @synchronize()
            if in_k_ok
                @unroll for i in 1:emb_dim
                    KA.@atomic dk[i, tidx + lo_k, kv_head_idx, gidx[2]] += d_shm[i,tidx]
                end
            end

            # -------------------- dQ ------------------------------------
            in_dq = in_seq_bounds || tidx + lo_q ≤ size(dq, 2)
            sh_load_emb!(d_shm, dq, lo_q, in_dq, Val(false), q_head_idx)
            @synchronize()
            mma!(d_shm, s_shm, k_shm, cfg_dq, tidx, mma_acc_fn)
            @synchronize()
            if in_dq
                @unroll for i in 1:emb_dim
                    dq[i, tidx + lo_q, q_head_idx, gidx[2]] = d_shm[i,tidx]
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
    @unroll for i in 1:emb_dim
        Δ_scaled_v[i] = Δ_v[i] * inv_denom
    end

    # δ = sum(o * do; dims=2) # dims=2 in the (B, H, L, E) format
    o_v = @view(o[:, tidx + q_offset, gidx[2], gidx[3]])
    d = zero(T)
    @unroll for i in 1:emb_dim
        d += Δ_scaled_v[i] * o_v[i]
    end
    δ[tidx + q_offset, gidx[2], gidx[3]] = d
end

function ∇flash_attention(
    Δ::AbstractArray{T,4},
    o::AbstractArray{T,4}, ms::AbstractArray{T,3}, ls::AbstractArray{T,3},
    q::AbstractArray{T,4}, k::AbstractArray{T,4}, v::AbstractArray{T,4},
    pair::Maybe{AbstractArray{T,4}} = nothing;
    causal::Bool,
    kpad_mask::Maybe{AbstractMatrix{Bool}} = nothing,
) where T
    emb_dim, QL, QH, B = size(q)
    KL, KVH = size(k, 2), size(k, 3)
    
    # Validate grouped-query attention
    @assert QH % KVH == 0 "Number of query heads ($QH) must be divisible by number of KV heads ($KVH)"
    num_q_per_kv = QH ÷ KVH

    kab          = get_backend(q)
    target_shmem = shared_memory(kab, KA.device(kab))
    gsz          = flash_attention_groupsize(T; emb_dim, target_shmem)

    q_tiles, k_tiles = cld.((QL, KL), gsz)
    in_bounds = QL % gsz == 0 && KL % gsz == 0
    scale     = T(inv(sqrt(emb_dim)))

    # ---------------- preprocess -----------------------------------------
    Δ_scaled = similar(Δ);  δ = similar(ls)
    threads  = (gsz,1,1);   ndrange = (gsz*q_tiles, QH, B)
    _flash_attention_bwd_preprocess!(kab, threads)(
        Δ_scaled, δ, Δ, o, ls,
        Val(emb_dim), Val(in_bounds); ndrange)

    # ---------------- output grads ---------------------------------------
    dq = KA.zeros(kab, T, size(q))
    dk = KA.zeros(kab, T, size(k))
    dv = KA.zeros(kab, T, size(v))
    dp = isnothing(pair) ?
        KA.allocate(kab, T, (0,0,0,0)) :   # harmless dummy
        KA.zeros(kab, T, size(pair))

    # ---------------- MMA configs (unchanged) ----------------------------
    BM,BK,BN = gsz, emb_dim, gsz
    TM,TN    = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg      = FATileConfig{BM,BK,BN,TM,TN,false,false,false}

    BM,BK,BN = emb_dim, gsz, gsz
    TM,TN    = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg_dv   = FATileConfig{BM,BK,BN,TM,TN,false,false,false}

    BM,BK,BN = gsz, gsz, emb_dim
    TM,TN    = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg_dk   = FATileConfig{BM,BK,BN,TM,TN,true,false,true}
    cfg_dq   = FATileConfig{BM,BK,BN,TM,TN,false,true,true}

    BM,BK,BN = gsz, emb_dim, gsz
    TM,TN    = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg_ds   = FATileConfig{BM,BK,BN,TM,TN,true,false,false}

    # ---------------- launch kernel --------------------------------------
    threads = (gsz, 1)
    ndrange = (gsz * QH, B)
    _flash_attention_bwd!(kab, threads)(
        cfg, cfg_dv, cfg_dk, cfg_dq, cfg_ds,
        dq, dk, dv, dp,
        Δ_scaled, δ,
        o, ms,
        q, k, v, scale,
        pair, kpad_mask,
        Val(emb_dim), Val(in_bounds), Val(causal), Val(num_q_per_kv);
        ndrange)

    return dq, dk, dv, (isnothing(pair) ? nothing : dp)
end
