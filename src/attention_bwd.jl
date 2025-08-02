@kernel unsafe_indices=true cpu=false inbounds=true function _flash_attention_bwd!(
    cfg, cfg_dv, cfg_dk, cfg_dq, cfg_ds,
    # ─────────── outputs ───────────
    dq::AbstractArray{T,4}, dk::AbstractArray{T,4}, dv::AbstractArray{T,4},
    # ─────────── inputs ────────────
    Δ::AbstractArray{T,4},
    δ::AbstractArray{T,3},
    o::AbstractArray{T,4},
    ms::AbstractArray{T,3},
    q::AbstractArray{T,4}, k::AbstractArray{T,4}, v::AbstractArray{T,4},
    scale::T,
    kpad_mask,                                    # NEW  (Bool(KL,B) or nothing)
    ::Val{emb_dim}, ::Val{q_seq_tiles}, ::Val{kv_seq_tiles},
    ::Val{in_seq_bounds}, ::Val{causal}, ::Val{use_padmask},   # NEW flag
) where {T, emb_dim, q_seq_tiles, kv_seq_tiles, in_seq_bounds, causal, use_padmask}
    gsz = @groupsize()[1]

    # shared-memory buffers (unchanged) ....................................
    q_shm = @localmem Float16 (gsz, emb_dim)
    k_shm = @localmem Float16 (emb_dim, gsz)
    s_shm = @localmem T      (gsz, gsz)
    Δ_shm = @localmem T      (emb_dim, gsz)
    d_shm = @localmem T      (emb_dim, gsz)

    tidx = @index(Local)
    gidx = @index(Group, NTuple)

    # small helper – identical to forward path
    @inline function sh_load_emb!(dest, src, offset, mask::Bool, ::Val{tr}) where tr
        @unroll for i in 1:emb_dim
            x, y = tr ? (tidx, i) : (i, tidx)
            @inbounds dest[x,y] = mask ? src[i, tidx+offset, gidx[1], gidx[2]] : zero(T)
        end
    end

    # main two-nested-tile loops ...........................................
    for start_n in 1:kv_seq_tiles
        lo  = (start_n - 1) * gsz             # column offset
        q_offset = causal ? lo : 0

        in_k_seq_bounds = in_seq_bounds || tidx + lo ≤ size(k,2)
        sh_load_emb!(k_shm, k, lo, in_k_seq_bounds, Val(false))
        @synchronize()

        start_iter = causal ? start_n : 1
        for start_m in start_iter:q_seq_tiles
            lo_inner = (start_m - 1) * gsz    # row offset

            in_q_seq_bounds = in_seq_bounds || tidx + q_offset ≤ size(q,2)
            sh_load_emb!(Δ_shm, Δ, q_offset, in_q_seq_bounds, Val(false))
            sh_load_emb!(q_shm, q, q_offset, in_q_seq_bounds, Val(true))
            @synchronize()

            # ----- recompute scores ---------------------------------------
            mma!(s_shm, q_shm, k_shm, cfg, tidx,
                 (res,_,__,___) -> res * scale)
            @synchronize()

            if causal
                @unroll for i in 1:gsz
                    (in_seq_bounds || i+lo ≤ size(k,2)) || break
                    s_shm[tidx,i] = ifelse(tidx+q_offset ≥ i+lo,
                                           s_shm[tidx,i], typemin(T))
                end
            end
            if use_padmask
                @unroll for i in 1:gsz
                    (in_seq_bounds || i+lo ≤ size(k,2)) || break
                    valid = @inbounds kpad_mask[i+lo, gidx[2]]
                    s_shm[tidx,i] = valid ? s_shm[tidx,i] : typemin(T)
                end
            end

            # ----- soft-max reconstruction --------------------------------
            in_ms_seq_bounds = in_seq_bounds || tidx + lo_inner ≤ size(ms,1)
            m_i = in_ms_seq_bounds ? ms[tidx+lo_inner, gidx[1], gidx[2]] : typemax(T)
            @unroll for i in 1:gsz
                s_shm[tidx,i] = exp(s_shm[tidx,i] - m_i)
            end

            # ----- dv update ----------------------------------------------
            in_dv_bounds = in_seq_bounds || tidx + lo ≤ size(dv,2)
            sh_load_emb!(d_shm, dv, lo, in_dv_bounds, Val(false))
            @synchronize()
            mma!(d_shm, Δ_shm, s_shm, cfg_dv, tidx, mma_acc_fn)
            @synchronize()
            if in_dv_bounds
                @unroll for i in 1:emb_dim
                    dv[i, tidx+lo, gidx[1], gidx[2]] = d_shm[i, tidx]
                end
            end

            # ----- ds computation -----------------------------------------
            sh_load_emb!(d_shm, v, lo, in_dv_bounds, Val(false))
            @synchronize()
            mma!(s_shm, Δ_shm, d_shm, cfg_ds, tidx,
                 (res, out, x, y) -> begin
                     d_i = (in_seq_bounds || x+lo_inner ≤ size(δ,1)) ?
                           @inbounds(δ[x+lo_inner, gidx[1], gidx[2]]) : zero(T)
                     out[x,y] * (res - d_i) * scale
                 end)

            # ----- dk update ----------------------------------------------
            sh_load_emb!(d_shm, dk, lo, in_k_seq_bounds, Val(false))
            @synchronize()
            mma!(d_shm, s_shm, q_shm, cfg_dk, tidx, mma_acc_fn)
            @synchronize()
            if in_k_seq_bounds
                @unroll for i in 1:emb_dim
                    dk[i, tidx+lo, gidx[1], gidx[2]] = d_shm[i, tidx]
                end
            end

            # ----- dq update ----------------------------------------------
            in_dq_bounds = in_seq_bounds || tidx + lo_inner ≤ size(dq,2)
            sh_load_emb!(d_shm, dq, lo_inner, in_dq_bounds, Val(false))
            @synchronize()
            mma!(d_shm, s_shm, k_shm, cfg_dq, tidx, mma_acc_fn)
            @synchronize()
            if in_dq_bounds
                @unroll for i in 1:emb_dim
                    dq[i, tidx+lo_inner, gidx[1], gidx[2]] = d_shm[i, tidx]
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
    Δ::AbstractArray{T,4},
    o::AbstractArray{T,4}, ms::AbstractArray{T,3}, ls::AbstractArray{T,3},
    q::AbstractArray{T,4}, k::AbstractArray{T,4}, v::AbstractArray{T,4};
    causal::Bool,
    kpad_mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
) where T
    emb_dim, QL, H, B = size(q)
    KL = size(k, 2)

    kab          = get_backend(q)
    target_shmem = shared_memory(kab, KA.device(kab))
    gsz          = flash_attention_groupsize(T; emb_dim, target_shmem)

    q_seq_tiles, kv_seq_tiles = cld.( (QL, KL), gsz )
    threads   = (gsz, 1)
    ndrange   = (gsz * H, B)
    in_bounds = QL % gsz == 0 && KL % gsz == 0
    use_mask  = kpad_mask !== nothing
    scale     = T(inv(sqrt(emb_dim)))

    dq = KA.zeros(kab, T, size(q))
    dk = KA.zeros(kab, T, size(k))
    dv = KA.zeros(kab, T, size(v))

    # --- MMA configs (exactly the originals) ------------------------------
    BM, BK, BN = gsz, emb_dim, gsz
    TM, TN     = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg        = FATileConfig{BM,BK,BN,TM,TN,false,false,false}

    BM, BK, BN = emb_dim, gsz, gsz
    TM, TN     = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg_dv     = FATileConfig{BM,BK,BN,TM,TN,false,false,false}

    BM, BK, BN = gsz, gsz, emb_dim
    TM, TN     = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg_dk     = FATileConfig{BM,BK,BN,TM,TN,true,false,true}

    cfg_dq     = cfg_dk  # same geometry
    BM, BK, BN = gsz, emb_dim, gsz
    TM, TN     = flash_attention_mma_thread_cfg(gsz; BM, BN)
    cfg_ds     = FATileConfig{BM,BK,BN,TM,TN,true,false,false}
    # ----------------------------------------------------------------------

    _flash_attention_bwd!(kab, threads)(
        cfg, cfg_dv, cfg_dk, cfg_dq, cfg_ds,
        dq, dk, dv,                 # outputs
        Δ,                          # Δ already scaled in kernel
        ls, o, ms,
        q, k, v, scale,
        kpad_mask,                  # NEW
        Val(emb_dim), Val(q_seq_tiles), Val(kv_seq_tiles),
        Val(in_bounds), Val(causal), Val(use_mask); ndrange)

    return dq, dk, dv
end
