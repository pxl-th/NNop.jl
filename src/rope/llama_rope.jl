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

@kernel cpu=false unsafe_indices=true inbounds=true function llama_rope!(
    q, k, cos, sin, sin_sign::Float32,
    ::Val{n_seq_tiles}, ::Val{half_dim},
) where {n_seq_tiles, half_dim}
    gsz = @groupsize()[1]
    idx = @index(Local)
    gid = @index(Group, NTuple)

    hi, bi = gid

    q_seq = size(q, 2)
    q_heads = size(q, 3)
    k_heads = size(k, 3)

    offset = 0
    @unroll for _ in 1:n_seq_tiles
        seq_idx = idx + offset
        seq_idx ≤ q_seq || continue

        @unroll for i in 1:half_dim
            other_i = i + half_dim
            c = cos[i, seq_idx, bi]
            s = sin[i, seq_idx, bi] * sin_sign

            if hi ≤ q_heads
                x1 = q[i, seq_idx, hi, bi]
                x2 = q[other_i, seq_idx, hi, bi]

                q[i, seq_idx, hi, bi]       = x1 * c - x2 * s
                q[other_i, seq_idx, hi, bi] = x2 * c + x1 * s
            end
            if hi ≤ k_heads
                x1 = k[i, seq_idx, hi, bi]
                x2 = k[other_i, seq_idx, hi, bi]

                k[i, seq_idx, hi, bi]       = x1 * c - x2 * s
                k[other_i, seq_idx, hi, bi] = x2 * c + x1 * s
            end
        end
        offset += gsz
    end
end

# q, k: [head dim, seq, n heads, batch]
# cos, sin: [dim, seq, batch]
function _llama_rope(q, k, cos, sin; bwd::Bool)
    @assert size(q, 1) == size(k, 1)
    @assert size(q, 2) == size(k, 2)
    @assert size(q, 4) == size(k, 4)

    kab = get_backend(q)
    q = copy(q)
    k = copy(k)

    half_dim = size(q, 1) ÷ 2
    q_heads, k_heads = size(q, 3), size(k, 3)

    gsz = 256
    ndrange = (gsz * max(q_heads, k_heads), size(q, 4))
    n_seq_tiles = cld(size(q, 2), gsz)

    llama_rope!(kab, gsz)(
        q, k, cos, sin, bwd ? -1f0 : 1f0,
        Val(n_seq_tiles), Val(half_dim); ndrange)
    return q, k
end

llama_rope(q, k; cos, sin) = _llama_rope(q, k, cos, sin; bwd=false)
∇llama_rope(dq, dk; cos, sin) = _llama_rope(dq, dk, cos, sin; bwd=true)

function CRC.rrule(::typeof(llama_rope), q, k; cos, sin)
    q, k = llama_rope(q, k; cos, sin)
    _pullback(Δ) = (CRC.NoTangent(), ∇llama_rope(CRC.unthunk.(Δ)...; cos, sin)...)
    return (q, k), _pullback
end
