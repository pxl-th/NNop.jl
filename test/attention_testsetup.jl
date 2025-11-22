@testsetup module TSAttention

export naive_softmax, att_padding_mask, naive_attention

import ChainRulesCore as CRC
using NNlib: ⊠, make_causal_mask, apply_attn_mask

function naive_softmax(x; dims = 1)
    mx = maximum(x; dims)
    tmp = exp.(x .- mx)
    return tmp ./ sum(tmp; dims)
end

function att_padding_mask(kpadmask, other_dim; T = Float32)
    pm = T.(kpadmask)
    return CRC.@ignore_derivatives log.(reshape(pm, size(pm,1), 1, 1, size(pm,2)))
end

function naive_attention(q, k, v, pair = nothing; causal::Bool, kpad_mask::Union{Nothing,AbstractMatrix{Bool}} = nothing)
    kt = permutedims(k, (2, 1, 3, 4))
    a = (kt ⊠ q) .* inv(sqrt(size(q, 1)))
    if causal
        m = make_causal_mask(q)
        a = apply_attn_mask(a, m)
    end
    if !isnothing(kpad_mask)
        a = a .+ att_padding_mask(kpad_mask, size(q, 2))
    end
    if !isnothing(pair)
        a = a .+ permutedims(pair, (3, 2, 1, 4)) #When it comes in as H-QL-KL-B
    end
    return v ⊠ naive_softmax(a; dims=1)
end

end
