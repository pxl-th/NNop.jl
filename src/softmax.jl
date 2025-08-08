struct MD
    m::Float32
    d::Float32
end

function md_reduce(a::MD, b::MD)::MD
    a_bigger = a.m > b.m
    bigger_m, smaller_m = a_bigger ? (a, b) : (b, a)
    diff = smaller_m.m - bigger_m.m
    # Avoid NaN in output.
    diff = ifelse(isnan(diff), -Inf32, diff)
    return MD(
        bigger_m.m,
        bigger_m.d + smaller_m.d * exp(diff),
    )
end


@kernel cpu=false inbounds=true function online_softmax!(
    y::AbstractMatrix{T}, x::AbstractMatrix{T}, ::Val{n_loop_iters}, ::Val{in_seq_bounds},
) where {T, n_loop_iters, in_seq_bounds}
    gsz = @groupsize()[1]
    N = size(x, 1)

    idx = @index(Local)
    bid = @index(Group)

    # Reposition to the current batch.
    xv = @view(x[:, bid])
    yv = @view(y[:, bid])

    # Each thread finds its own maximum(x; dims=bid) & calculates respective denominator.
    md_partial = MD(-Inf32, 0f0)
    elem_id = idx
    @unroll for i in 1:n_loop_iters
        if in_seq_bounds || elem_id ≤ size(x, 1)
            md_partial = md_reduce(md_partial, MD(xv[elem_id], 1f0))
        end
        elem_id += gsz
    end

    # Reduce to compute total max & total denominator values.
    md = @groupreduce md_reduce md_partial
    md_total = @localmem MD 1
    idx == 1 && (md_total[1] = md)
    @synchronize()

    total_max = md_total[1].m
    inv_denom = inv(md_total[1].d)

    elem_id = idx
    @unroll for i in 1:n_loop_iters
        if in_seq_bounds || elem_id ≤ size(x, 1)
            yv[elem_id] = exp(xv[elem_id] - total_max) * inv_denom
        end
        elem_id += gsz
    end
end

function online_softmax(x::T)::T where T <: AbstractMatrix
    y = similar(x)
    gsz = 256
    n_loop_iters = ceil(Int, size(x, 1) / gsz)
    in_seq_bounds = size(x, 1) % gsz == 0
    ndrange = gsz * size(x, 2)
    online_softmax!(get_backend(x), gsz)(y, x, Val(n_loop_iters), Val(in_seq_bounds); ndrange)
    return y
end

function ∇online_softmax(Δ::AbstractArray{T}, y::AbstractArray{S}) where {T, S}
    dx = if within_gradient(y)
        tmp = Δ .* y
        tmp .- y .* sum(tmp; dims=1)
    else
        # This path is faster, only safe for 1st derivatives though.
        out = similar(y, promote_type(T, S))
        out .= Δ .* y
        out .= out .- y .* sum(out; dims=1)
    end
end

function CRC.rrule(::typeof(online_softmax), x)
    y = online_softmax(x)
    _pullback(Δ) = (CRC.NoTangent(), ∇online_softmax(CRC.unthunk(Δ), y))
    return y, _pullback
end
