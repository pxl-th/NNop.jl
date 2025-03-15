# TODO
# - boundschecking
# - bwd pass (reuse from NNlib.jl)
# - CRC integration

struct MD
    m::Float32
    d::Float32
end

function md_reduce(a::MD, b::MD)::MD
    a_bigger = a.m > b.m
    bigger_m, smaller_m = a_bigger ? (a, b) : (b, a)
    return MD(
        bigger_m.m,
        bigger_m.d + smaller_m.d * exp(smaller_m.m - bigger_m.m),
    )
end

@kernel cpu=false inbounds=true function online_softmax!(
    y::AbstractMatrix{T}, x::AbstractMatrix{T}, ::Val{tile_size}, ::Val{n_loop_iters},
) where {T, tile_size, n_loop_iters}
    gsz::Int = prod(@groupsize())

    idx = @index(Local)
    bid = @index(Group)

    # Reposition to the current batch.
    xv = @view(x[:, bid])
    yv = @view(y[:, bid])

    # Each thread finds its own maximum(x; dims=bid) & calculates respective denominator.
    md_partial = MD(-Inf32, 0f0)
    elem_id = idx
    @unroll for i in 1:n_loop_iters
        # Each thread loads & accumulates `tile_size` adjacent elements.
        ptr = pointer(xv, (elem_id - 1) * tile_size + 1)
        xv_tile = vload(SIMD.Vec{tile_size, T}, ptr)
        @unroll for j in 1:tile_size
            md_partial = md_reduce(md_partial, MD(xv_tile[j], 1f0))
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
        tile_idx = (elem_id - 1) * tile_size + 1

        xv_ptr = pointer(xv, tile_idx)
        xv_tile = vload(SIMD.Vec{tile_size, T}, xv_ptr)
        xv_tile = exp(xv_tile - total_max) * inv_denom

        yv_ptr = pointer(yv, tile_idx)
        vstore!(yv_ptr, xv_tile)

        elem_id += gsz
    end
end

function online_softmax(x::T)::T where T <: AbstractMatrix
    tile_size = 4
    @assert size(x, 1) % (256 * tile_size ) == 0
    n_loop_iters = ceil(Int, size(x, 1) / (256 * tile_size))
    y = similar(x)
    online_softmax!(get_backend(x), 256)(
        y, x, Val(tile_size), Val(n_loop_iters);
        ndrange=256 * size(x, 2))
    return y
end
