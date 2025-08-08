struct C12
    c1::Float32
    c2::Float32
end

C12_reduce(a::C12, b::C12)::C12 = C12(a.c1 + b.c1, a.c2 + b.c2)

@kernel cpu=false unsafe_indices=true inbounds=true function _layer_norm!(
    y, μ, Σ,
    x, w, b, ϵ::Float32,
    inv_emb_size::Float32,
    ::Val{in_seq_bounds}, ::Val{n_loop_iters},
) where {in_seq_bounds, n_loop_iters}
    gsz = @groupsize()[1]
    idx = @index(Local)
    bid = @index(Group)

    xv = @view(x[:, bid])
    yv = @view(y[:, bid])

    # Compute mean.
    μi::Float32 = 0f0
    elem_id = idx
    @unroll for _ in 1:n_loop_iters
        if in_seq_bounds || elem_id ≤ size(x, 1)
            μi += Float32(xv[elem_id])
        end
        elem_id += gsz
    end
    μi *= inv_emb_size
    μi = @groupreduce(+, μi)

    idx == 1 && (μ[bid] = μi)
    @synchronize()
    μi = μ[bid]

    # Compute variance.
    σ::Float32 = 0f0
    elem_id = idx
    @unroll for _ in 1:n_loop_iters
        if in_seq_bounds || elem_id ≤ size(x, 1)
            xc = Float32(xv[elem_id]) - μi
            σ += xc * xc
        end
        elem_id += gsz
    end
    σ *= inv_emb_size
    σ = @groupreduce(+, σ)

    idx == 1 && (Σ[bid] = inv(sqrt(σ + ϵ)))
    @synchronize()
    σ = Σ[bid]

    # Compute layer normalization.
    elem_id = idx
    @unroll for _ in 1:n_loop_iters
        if in_seq_bounds || elem_id ≤ size(x, 1)
            xc = Float32(xv[elem_id]) - μi
            yv[elem_id] = xc * σ * Float32(w[elem_id]) + Float32(b[elem_id])
        end
        elem_id += gsz
    end
end

@kernel cpu=false unsafe_indices=true inbounds=true function _∇layer_norm!(
    dx, dw, db,
    Δ, μ, Σ,
    x, w, b,
    inv_emb_size::Float32, batches_per_groupsize::Int,
    ::Val{emb}, ::Val{in_seq_bounds}, ::Val{n_loop_iters},
) where {emb, in_seq_bounds, n_loop_iters}
    n = size(x, 2)

    gsz = @groupsize()[1]
    idx = @index(Local)
    bid = @index(Group)

    dw_sh = @localmem Float32 emb
    db_sh = @localmem Float32 emb
    c_sh = @localmem C12 1

    elem_id = idx
    @unroll for _ in 1:n_loop_iters
        if in_seq_bounds || elem_id ≤ emb
            dw_sh[elem_id] = 0f0
            db_sh[elem_id] = 0f0
        end
        elem_id += gsz
    end
    @synchronize()

    ns = (bid - 1) * batches_per_groupsize + 1
    ne = min(n, bid * batches_per_groupsize)
    for i in ns:ne
        μi, σ = μ[i], Σ[i]

        # Compute `c1 = wdy ⋅ xn` & `c2 = sum(wdy)`.
        elem_id = idx
        c1::Float32, c2::Float32 = 0f0, 0f0
        @unroll for _ in 1:n_loop_iters
            if in_seq_bounds || elem_id ≤ emb
                xi, wi, δ = Float32(x[elem_id, i]), Float32(w[elem_id]), Float32(Δ[elem_id, i])
                xn = (xi - μi) * σ
                wdy = wi * δ
                c1 += wdy * xn
                c2 += wdy
            end
            elem_id += gsz
        end
        c1 *= inv_emb_size
        c2 *= inv_emb_size

        c12 = C12(c1, c2)
        c12 = @groupreduce(C12_reduce, c12)
        idx == 1 && (c_sh[1] = c12)
        @synchronize()

        c12 = c_sh[1]
        c1, c2 = c12.c1, c12.c2

        # Compute `dx, dw, db`.
        elem_id = idx
        @unroll for _ in 1:n_loop_iters
            if in_seq_bounds || elem_id ≤ emb
                xi, wi, δ = Float32(x[elem_id, i]), Float32(w[elem_id]), Float32(Δ[elem_id, i])
                xn = (xi - μi) * σ
                wdy = wi * δ
                dxi = (wdy - (xn * c1 + c2)) * σ
                dwi = δ * xn

                dx[elem_id, i] = dxi
                dw_sh[elem_id] += dwi
                db_sh[elem_id] += δ
            end
            @synchronize()
            elem_id += gsz
        end
    end

    elem_id = idx
    @unroll for _ in 1:n_loop_iters
        if in_seq_bounds || elem_id ≤ emb
            dw[bid, elem_id] = dw_sh[elem_id]
            db[bid, elem_id] = db_sh[elem_id]
        end
        elem_id += gsz
    end
end

function _layer_norm(x, w, b; ϵ::Float32 = 1f-6)
    kab = get_backend(x)

    emb, n = size(x)
    y = similar(x)
    μ = KA.allocate(kab, Float32, n)
    Σ = KA.allocate(kab, Float32, n)

    gsz = 256
    ndrange = gsz * n
    inv_emb_size = 1f0 / Float32(emb)

    in_seq_bounds = size(x, 1) % gsz == 0
    n_loop_iters = ceil(Int, emb / gsz)

    _layer_norm!(kab, gsz)(
        y, μ, Σ,
        x, w, b, ϵ,
        inv_emb_size, Val(in_seq_bounds), Val(n_loop_iters); ndrange)
    return y, μ, Σ
end

function ∇layer_norm(Δ, μ, Σ, x, w, b)
    emb, n = size(x)
    batches_per_groupsize = 4 # TODO == sm_count, query
    n_batch_loop_iters = ceil(Int, n / batches_per_groupsize)

    kab = get_backend(x)
    dx = similar(x)
    dw = KA.allocate(kab, eltype(w), (n_batch_loop_iters, emb))
    db = KA.allocate(kab, eltype(b), (n_batch_loop_iters, emb))

    gsz = 256
    ndrange = gsz * n_batch_loop_iters
    inv_emb_size = 1f0 / Float32(emb)

    in_seq_bounds = size(x, 1) % gsz == 0
    n_loop_iters = ceil(Int, emb / gsz)
    emb_sh_size = emb

    _∇layer_norm!(kab, gsz)(
        dx, dw, db,
        Δ, μ, Σ,
        x, w, b,
        inv_emb_size, batches_per_groupsize,
        Val(emb_sh_size), Val(in_seq_bounds), Val(n_loop_iters); ndrange)

    dw, db = if n_batch_loop_iters == 1
        reshape(dw, :), reshape(db, :)
    else
        # TODO dedicated kernel
        reshape(sum(dw; dims=1), :), reshape(sum(db; dims=1), :)
    end
    return dx, dw, db
end

function layer_norm(x, w, b; ϵ::Float32 = 1f-6)
    y = _layer_norm(x, w, b; ϵ)
    within_gradient(x) && return y
    @assert length(y) == 3
    return y[1]
end

function CRC.rrule(::typeof(_layer_norm), x, w, b; ϵ::Float32 = 1f-6)
    y, μ, Σ = _layer_norm(x, w, b; ϵ)
    function _pullback(Δ)
        dx, dw, db = ∇layer_norm(CRC.unthunk(Δ), μ, Σ, x, w, b)
        return CRC.NoTangent(), dx, dw, db
    end
    return y, _pullback
end
