@kernel cpu=false inbounds=true function _flash_attention_fwd!(
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
        # update `k_shm`
        for i in 1:emb_dim
            k_shm[i, tidx] = k[i, tidx + k_offset, gidx[2], gidx[3]]
        end
        @synchronize()

        # compute q' * k (L_q, L_k)
        # each thread computes [tidx, :] * [:, :]
        for i in 1:gsz
            tmp = zero(T)
            for j in 1:emb_dim
                tmp += q_shm[tidx, j] * k_shm[j, i]
            end
            s_shm[tidx, i] = tmp
            # TODO use when causal
            # s_shm[tidx, i] = ifelse(tidx + q_offset >= i + k_offset, tmp, typemin(T))
        end
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

        # apply `p_scale` to `s_shm`
        for i in 1:gsz
            s_shm[tidx, i] *= p_scale
        end
        # apply `o_scale` to `o_shm`
        for i in 1:emb_dim
            o_shm[i, tidx] *= o_scale
        end
        # load `v` into `k_shm` (shared shmem).
        for i in 1:emb_dim
            k_shm[i, tidx] = v[i, tidx + k_offset, gidx[2], gidx[3]]
        end
        @synchronize()

        # (q' * k) * v' (L_q, emb_dim)
        for i in 1:emb_dim
            tmp = zero(T)
            for j in 1:gsz
                tmp += s_shm[tidx, j] * k_shm[i, j]
            end
            o_shm[i, tidx] += tmp
        end
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

function flash_attention(
    q::AbstractArray{T, 4}, k::AbstractArray{T, 4}, v::AbstractArray{T, 4},
) where T
    emb_dim, L, H, N = size(q)

    gsz = min(64, L)
    n_seq_tiles = cld(L, gsz)
    threads = (gsz, 1, 1)
    ndrange = (gsz * n_seq_tiles, H, N)

    kab = get_backend(q)
    o = similar(q)
    ms = KA.allocate(kab, eltype(o), (L, H, N))
    ls = KA.allocate(kab, eltype(o), (L, H, N))

    _flash_attention_fwd!(kab, threads)(
        o, ms, ls,
        q, k, v,
        Val(emb_dim), Val(n_seq_tiles);
        ndrange)

    return o, ms, ls
end

function ∇flash_attention(
    Δ::AbstractArray{T, 4},
    o::AbstractArray{T, 4}, ms::AbstractArray{T, 3}, ls::AbstractArray{T, 3},
    q::AbstractArray{T, 4}, k::AbstractArray{T, 4}, v::AbstractArray{T, 4},
) where T
    emb_dim, L, H, N = size(q)

    gsz = min(64, L)
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
    _flash_attention_bwd!(kab, threads)(
        # Output.
        dq, dk, dv,
        # Input.
        Δ_scaled, δ,
        o, ms, ls,
        q, k, v,
        Val(emb_dim), Val(n_seq_tiles); ndrange)

    return dq, dk, dv
end

@kernel cpu=false inbounds=true function _flash_attention_bwd!(
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

    q_shm = @localmem T (gsz, emb_dim)
    k_shm = @localmem T (emb_dim, gsz)
    s_shm = @localmem T (gsz, gsz)

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
                q_shm[tidx, i] = q[i, tidx + q_offset, gidx[1], gidx[2]]
            end
            @synchronize()

            # recompute q' * k (L_q, L_k)
            # each thread computes [tidx, :] * [:, :]
            for i in 1:gsz
                tmp = zero(T)
                for j in 1:emb_dim
                    tmp += q_shm[tidx, j] * k_shm[j, i]
                end
                s_shm[tidx, i] = tmp
                # TODO use when causal
                # s_shm[tidx, i] = ifelse(i + lo_inner >= tidx + lo, tmp, typemin(T))
            end
            @synchronize()

            m_i = ms[tidx + lo_inner, gidx[1], gidx[2]]
            for i in 1:gsz
                s_shm[tidx, i] = exp(s_shm[tidx, i] - m_i)
            end
            @synchronize()

            # compute dv += Δ (emb_dim, L_q) * s_shm (L_q, L_k) = (emb_dim, L_k)
            for i in 1:emb_dim
                tmp = zero(T)
                for j in 1:gsz
                    tmp += s_shm[j, tidx] * Δ[i, j + q_offset, gidx[1], gidx[2]]
                end
                dv[i, tidx + lo, gidx[1], gidx[2]] += tmp
            end
            @synchronize()

            # compute:
            # - dp = O - ls + dot(Δ, v)
            # - ds = p == s_shm (L_q, L_k) * dp
            d_i = δ[tidx + lo_inner, gidx[1], gidx[2]]
            for i in 1:gsz
                tmp = zero(T)
                for j in 1:emb_dim
                    tmp += Δ[j, tidx + q_offset, gidx[1], gidx[2]] * v[j, i + lo, gidx[1], gidx[2]]
                end
                s_shm[tidx, i] *= tmp - d_i
            end
            @synchronize()

            # compute dk = dot(ds.T, q) (L_k, L_q) * (L_q, emb_dim)
            for i in 1:emb_dim
                tmp = zero(T)
                for j in 1:gsz
                    tmp += s_shm[j, tidx] * q_shm[j, i]
                end
                dk[i, tidx + lo, gidx[1], gidx[2]] += tmp
            end

            # compute dq = dot(ds, k) (L_q, L_k) * (L_k, emb_dim)
            for i in 1:emb_dim
                tmp = zero(T)
                for j in 1:gsz
                    tmp += s_shm[tidx, j] * k_shm[i, j]
                end
                dq[i, tidx + lo_inner, gidx[1], gidx[2]] += tmp
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

function naive_attention(q, k, v)
    kt = permutedims(k, (2, 1, 3, 4))
    a = kt ⊠ q
    am = maximum(a; dims=1)
    return v ⊠ naive_softmax(a .- am; dims=1)
end
