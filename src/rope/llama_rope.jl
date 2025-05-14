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
    q, k, cos, sin,
    ::Val{n_seq_tiles}, ::Val{q_half_dim}, ::Val{k_half_dim}, ::Val{bwd},
) where {n_seq_tiles, q_half_dim, k_half_dim, bwd}
    gsz = @groupsize()[1]
    idx = @index(Local)
    gid = @index(Group, NTuple)

    sin_sign = ifelse(bwd, -1f0, 1f0)
    q_dim = size(q, 1)
    k_dim = size(k, 1)

    offset = 0
    for _ in 1:n_seq_tiles
        seq_idx = idx + offset

        @unroll for i in 1:q_half_dim
            other_i = (i - 1 + q_half_dim) % q_dim + 1

            x1 = q[i, seq_idx, gid[1], gid[2]]
            x2 = q[other_i, seq_idx, gid[1], gid[2]]

            c = cos[i, seq_idx, gid[2]]
            s = sin[i, seq_idx, gid[2]] * sin_sign

            q[i, seq_idx, gid[1], gid[2]]       = x1 * c - x2 * s
            q[other_i, seq_idx, gid[1], gid[2]] = x1 * c + x1 * s
        end
        @unroll for i in 1:k_half_dim
            other_i = (i - 1 + k_half_dim) % k_dim + 1

            x1 = k[i, seq_idx, gid[1], gid[2]]
            x2 = k[other_i, seq_idx, gid[1], gid[2]]

            c = cos[i, seq_idx, gid[2]]
            s = sin[i, seq_idx, gid[2]] * sin_sign

            k[i, seq_idx, gid[1], gid[2]]       = x1 * c - x2 * s
            k[other_i, seq_idx, gid[1], gid[2]] = x1 * c + x1 * s
        end
        offset += gsz
    end
end

# q, k: [head dim, seq, n heads, batch]
# cos, sin: [dim, seq, batch]
function _llama_rope(q, k, cos, sin; bwd::Bool)
    @assert size(cos) == size(sin)
    kab = get_backend(q)

    q = copy(q)
    k = copy(k)

    q_half_dim = size(q, 1) ÷ 2
    k_half_dim = size(k, 1) ÷ 2
    q_seq = size(q, 2)
    k_seq = size(k, 2)
    @assert q_seq == k_seq

    gsz = 256
    q_seq_tiles = cld(q_seq, gsz)
    k_seq_tiles = cld(k_seq, gsz)
    ndrange = (gsz * size(q, 3), size(q, 4))

    llama_rope!(kab, gsz)(
        q, k, cos, sin,
        Val(q_seq_tiles), # TODO kkkkk
        Val(q_half_dim),
        Val(k_half_dim),
        Val(bwd); ndrange)

    return q, k
end

# TODO
# - arbitrary q & k seq lengths

llama_rope(q, k; cos, sin) = _llama_rope(q, k, cos, sin; bwd=false)
∇llama_rope(dq, dk; cos, sin) = _llama_rope(dq, dk, cos, sin; bwd=true)

function ChainRulesCore.rrule(::typeof(llama_rope), q, k; cos, sin)
    q, k = llama_rope(q, k; cos, sin)
    _pullback(Δ) = (
        ChainRulesCore.NoTangent(),
        ∇llama_rope(ChainRulesCore.unthunk.(Δ)...; cos, sin)...)
    return (q, k), _pullback
end
