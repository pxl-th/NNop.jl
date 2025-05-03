@kernel function _rms_norm!(
    y, rms, x, w,
    inv_emb_size::Float32, ϵ::Float32,
    n_loop_iters::Int,
)
    gsz = @groupsize()[1]
    idx = @index(Local)
    bid = @index(Group)

    # Reposition to the current batch.
    xv = @view(x[:, bid])
    yv = @view(y[:, bid])

    partial::Float32 = 0f0
    elem_id = idx
    for i in 1:n_loop_iters
        x = xv[elem_id]
        partial += x * x
        elem_id += gsz
    end
    partial *= inv_emb_size

    avg = @groupreduce(+, partial)
    rstd = inv(sqrt(avg + ϵ))
    idx == 1 && (rms[bid] = rstd)
    @synchronize()
    rstd = rms[bid] # Reload to all threads.

    elem_id = idx
    for i in 1:n_loop_iters
        yv[elem_id] = w[elem_id] * xv[elem_id] * rstd
        elem_id += gsz
    end
end

# TODO offset
function rms_norm(x::AbstractMatrix, w::AbstractVector; ϵ::Float32)
    emb, n = size(x)
    @assert emb == length(w)

    kab = get_backend(x)
    y = similar(x)
    rstd = KA.zeros(kab, Float32, (n,)) # for backward

    gsz = 256
    ndrange = gsz * n
    inv_emb_size = 1f0 / Float32(emb)

    in_seq_bounds = size(x, 1) % gsz == 0
    n_loop_iters = ceil(Int, emb / gsz)

    @show in_seq_bounds
    @show n_loop_iters
    @show ndrange

    _rms_norm!(kab, gsz)(
        y, rstd, x, w,
        inv_emb_size, ϵ,
        n_loop_iters; ndrange)
    return y
end
