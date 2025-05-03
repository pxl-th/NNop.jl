# TODO rounding modes

@kernel cpu=false unsafe_indices=true inbounds=true function _rms_norm!(
    y, rms, x, w, offset::Float32,
    inv_emb_size::Float32, ϵ::Float32,
    ::Val{in_seq_bounds}, ::Val{n_loop_iters},
) where {in_seq_bounds, n_loop_iters}
    gsz = @groupsize()[1]
    idx = @index(Local)
    bid = @index(Group)

    # Reposition to the current batch.
    xv = @view(x[:, bid])
    yv = @view(y[:, bid])

    partial::Float32 = 0f0
    elem_id = idx
    @unroll for _ in 1:n_loop_iters
        if in_seq_bounds || elem_id ≤ size(x, 1)
            partial += Float32(xv[elem_id])^2
        end
        elem_id += gsz
    end
    partial *= inv_emb_size
    avg = @groupreduce(+, partial)

    idx == 1 && (rms[bid] = inv(sqrt(avg + ϵ)))
    @synchronize()
    rstd = rms[bid] # Reload to all threads.

    elem_id = idx
    @unroll for _ in 1:n_loop_iters
        if in_seq_bounds || elem_id ≤ size(x, 1)
            yv[elem_id] = (offset + Float32(w[elem_id])) * Float32(xv[elem_id]) * rstd
        end
        elem_id += gsz
    end
end

# dd = (dy * (w + offset)) ⋅ x
# dx = (1 / RMS) * (dd * x * dy * (w + offset - (1 / N) * (1 / RMS²))
# dw = sum(dy * (x / RMS)) - summation over n_batch_loop_iters
@kernel cpu=false unsafe_indices=true inbounds=true function _∇rms_norm!(
    dx::AbstractMatrix, dw::AbstractMatrix,
    Δ::AbstractMatrix, rms::AbstractVector, x::AbstractMatrix, w::AbstractVector,
    offset::Float32, inv_emb_size::Float32, batches_per_groupsize::Int,
    ::Val{emb_sh_size}, ::Val{in_seq_bounds}, ::Val{n_loop_iters},
) where {emb_sh_size, in_seq_bounds, n_loop_iters}
    n = size(x, 2)

    gsz = @groupsize()[1]
    idx = @index(Local)
    bid = @index(Group)

    dw_sh = @localmem Float32 emb_sh_size
    dd_sh = @localmem Float32 1
    dd::Float32 = 0f0 # Define at the top to avoid weird divergence.

    # Init shmem to 0.
    elem_id = idx
    @unroll for _ in 1:n_loop_iters
        dw_sh[elem_id] = 0f0
        elem_id += gsz
    end
    @synchronize()

    ns = (bid - 1) * batches_per_groupsize + 1
    ne = min(n, bid * batches_per_groupsize)
    for i in ns:ne
        # Init shmem to 0 (for every batch).
        idx == 1 && (dd_sh[1] = 0f0)
        @synchronize()

        # Compute `dd = (dy * (w + offset)) ⋅ x`.
        dd = 0f0 # Reset to 0.
        elem_id = idx
        @unroll for _ in 1:n_loop_iters
            δ, w_i, x_i = in_seq_bounds || elem_id ≤ size(x, 1) ?
                (Float32(Δ[elem_id, i]), Float32(w[elem_id]), Float32(x[elem_id, i])) :
                (0f0, 0f0, 0f0)
            dd += δ * (w_i + offset) * x_i
            elem_id += gsz
        end

        dd_red = @groupreduce(+, dd)
        idx == 1 && (dd_sh[1] += dd_red)
        @synchronize()

        dd = dd_sh[1] # Reload to all threads.
        rstd = rms[i]
        elem_id = idx
        @unroll for _ in 1:n_loop_iters
            δ, w_i, x_i = in_seq_bounds || elem_id ≤ size(x, 1) ?
                (Float32(Δ[elem_id, i]), Float32(w[elem_id]), Float32(x[elem_id, i])) :
                (0f0, 0f0, 0f0)

            # Compute `dx`.
            m = δ * (w_i + offset)
            dx_i = rstd * m + rstd * (-inv_emb_size * rstd^2 * dd * x_i)
            # Compute `dw = sum(dy * (x / RMS))`, sum over `batches_per_groupsize`.
            dw_i = δ * (x_i * rstd)

            dw_sh[elem_id] += dw_i
            @synchronize()
            if in_seq_bounds || elem_id ≤ size(x, 1)
                dx[elem_id, i] = dx_i
            end
            elem_id += gsz
        end
    end

    elem_id = idx
    @unroll for _ in 1:n_loop_iters
        dw_i = dw_sh[elem_id]
        if in_seq_bounds || elem_id ≤ size(x, 1)
            dw[bid, elem_id] = dw_i
        end
        elem_id += gsz
    end
end

function _rms_norm(x::AbstractMatrix, w::AbstractVector; ϵ::Float32, offset::Float32 = 0f0)
    emb, n = size(x)
    @assert emb == length(w)

    kab = get_backend(x)
    y = similar(x)
    rms = KA.zeros(kab, Float32, (n,)) # cache for backward

    gsz = 256
    ndrange = gsz * n
    inv_emb_size = 1f0 / Float32(emb)

    in_seq_bounds = size(x, 1) % gsz == 0
    n_loop_iters = ceil(Int, emb / gsz)

    _rms_norm!(kab, gsz)(
        y, rms, x, w, offset,
        inv_emb_size, ϵ,
        Val(in_seq_bounds), Val(n_loop_iters); ndrange)
    return y, rms
end

function ∇rms_norm(Δ, rms, x, w; offset::Float32)
    emb, n = size(x)
    batches_per_groupsize = 4 # TODO == sm_count, query
    n_batch_loop_iters = ceil(Int, n / batches_per_groupsize)

    kab = get_backend(x)
    dx = KA.allocate(kab, eltype(x), size(x))
    dw = KA.allocate(kab, Float32, (n_batch_loop_iters, emb))

    gsz = 256
    ndrange = gsz * n_batch_loop_iters
    inv_emb_size = 1f0 / Float32(emb)

    in_seq_bounds = size(x, 1) % gsz == 0
    n_loop_iters = ceil(Int, emb / gsz)
    emb_sh_size = cld(emb, gsz) * gsz

    _∇rms_norm!(kab, gsz)(
        dx, dw,
        Δ, rms, x, w,
        offset, inv_emb_size, batches_per_groupsize,
        Val(emb_sh_size), Val(in_seq_bounds), Val(n_loop_iters); ndrange)

    dw = if n_batch_loop_iters == 1
        reshape(dw, :)
    else
        # TODO dedicated kernel that reduces over `sm_count`
        reshape(sum(dw; dims=1), :)
    end
    return dx, dw
end

function rms_norm(x, w; ϵ::Float32 = 1f-6, offset::Float32 = 0f0)
    y = _rms_norm(x, w; ϵ, offset)
    within_gradient(x) && return y
    @assert length(y) == 2
    return y[1]
end

function ChainRulesCore.rrule(::typeof(_rms_norm), x, w; ϵ::Float32 = 1f-6, offset::Float32 = 0f0)
    y, rms = _rms_norm(x, w; ϵ, offset)
    function _pullback(Δ)
        dx, dw = ∇rms_norm(ChainRulesCore.unthunk(Δ), rms, x, w; offset)
        return ChainRulesCore.NoTangent(), dx, dw
    end
    return y, _pullback
end
