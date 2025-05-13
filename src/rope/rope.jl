struct LlamaRotaryEmbedding{F <: AbstractVector{Float32}}
    inv_freq::F
    dim::Int
    base::Int
end

function LlamaRotaryEmbedding(dim::Int; base::Int = 10000)
    ids = (0f0:2f0:Float32(dim - 1)) ./ Float32(dim)
    inv_freq = inv.(base.^ids)
    LlamaRotaryEmbedding(inv_freq, dim, base)
end

# position_ids: (seq, batch)
# cos, sin: [dim, seq, batch]
function (emb::LlamaRotaryEmbedding)(position_ids::AbstractMatrix{Float32})
    # [1, seq, batch]
    position_ids = reshape(position_ids, 1, size(position_ids)...)
    # [dim] * [1, seq, batch] -> [dim, seq, batch]
    freqs = emb.inv_freq .* position_ids
    freqs = vcat(freqs, freqs)
    return cos.(freqs), sin.(freqs)
end

@kernel cpu=false function llama_rope!(q, k, cos, sin)
    idx = @index(Local)
    gid = @index(Group)

    # q * cos + rot_half(q) * sin
    q_dim = size(q, 1)
    k_dim = size(k, 1)
    q_half_dim = q_dim ÷ 2
    k_half_dim = q_dim ÷ 2

    # TODO unroll?
    for i in 1:q_half_dim
        other_i = (i - 1 + q_half_dim) % q_dim + 1
        x1 = q[i, idx, gid]
        x2 = q[other_i, idx, gid]

        q[i, idx, gid]       = x1 * cos[i, gid] - x2 * sin[i, gid]
        q[other_i, idx, gid] = x1 * cos[i, gid] + x1 * sin[i, gid]
    end
    for i in 1:k_half_dim
        other_i = (i - 1 + k_half_dim) % k_dim + 1
        x1 = k[i, idx, gid]
        x2 = k[other_i, idx, gid]

        k[i, idx, gid]       = x1 * cos[i, gid] - x2 * sin[i, gid]
        k[other_i, idx, gid] = x1 * cos[i, gid] + x1 * sin[i, gid]
    end
end

function llama_rope(q, k, cos, sin)
    @assert size(cos) == size(sin)
    kab = get_backend(q)

    # [head dim, seq, n heads, batch] -> [head dim, n heads, seq, batch]
    q = permutedims(q, (1, 3, 2, 4))
    k = permutedims(k, (1, 3, 2, 4))

    q_heads, k_heads = size(q, 2), size(k, 2)
    q_heads, k_heads = nextpow(2, q_heads), nextpow(2, k_heads)
    gsz = max(q_heads, k_heads)
    ndrange = gsz * prod(size(q)[3:4])
    @show gsz
    @show ndrange

    llama_rope!(kab, gsz)(
        reshape(q, size(q)[1:2]..., :),
        reshape(k, size(k)[1:2]..., :),
        reshape(cos, size(cos, 1), :),
        reshape(sin, size(sin, 1), :); ndrange)

    q = permutedims(q, (1, 3, 2, 4))
    k = permutedims(k, (1, 3, 2, 4))
    return q, k
end
