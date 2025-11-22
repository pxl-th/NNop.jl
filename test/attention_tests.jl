@testitem "Flash Attention" setup=[TSCore] begin

import Adapt
import Zygote
using NNlib: ⊠, make_causal_mask, apply_attn_mask

function naive_softmax(x; dims = 1)
    mx = maximum(x; dims)
    tmp = exp.(x .- mx)
    return tmp ./ sum(tmp; dims)
end

function att_padding_mask(kpadmask, other_dim; T = Float32)
    pm = T.(kpadmask)
    m = NNop.CRC.@ignore_derivatives log.(reshape(pm, size(pm,1), 1, 1, size(pm,2)) .* (similar(pm, 1, other_dim, 1, size(pm,2)) .= 1))
    return m
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

@testset "causal=$causal, padmask=$use_padmask, pair=$use_pair, T=$T, E=$E, QL=$QL, KL=$KL" for causal in (
    false, true
), use_padmask in (
    false, true,
), use_pair in (
    false, true,
), T in (
    Float32, # TODO more types
), E in (
    16, 32, 64, # TODO test on higher if applicable
), QL in (
    255, 256, 511, 512, 1024,
), KL in (
    255, 256, 511, 512, 1024,
)
    causal && QL != KL && continue

    H, B = 2, 3

    q = Adapt.adapt(kab, randn(T, E, QL, H, B))
    k = Adapt.adapt(kab, randn(T, E, KL, H, B))
    v = Adapt.adapt(kab, randn(T, E, KL, H, B))

    kpad_mask = nothing
    if use_padmask
        kpad_mask = Adapt.adapt(kab, ones(Bool, KL, B))
        kpad_mask[end-10:end, end] .= false
    end

    pair = nothing
    if use_pair
        pair = Adapt.adapt(kab, randn(T, H, QL, KL, B))
    end

    o1, ∇1 = Zygote.withgradient(q, k, v, pair) do q, k, v, pair
        sum(naive_attention(q, k, v, pair; causal, kpad_mask))
    end
    o2, ∇2 = Zygote.withgradient(q, k, v, pair) do q, k, v, pair
        sum(NNop.flash_attention(q, k, v, pair; causal, kpad_mask))
    end
    @test isapprox(o1, o2; atol=1e-3, rtol=1e-3)
    @test isapprox(∇1[1], ∇2[1]; atol=1e-3, rtol=1e-3)
    @test isapprox(∇1[2], ∇2[2]; atol=1e-3, rtol=1e-3)
    @test isapprox(∇1[3], ∇2[3]; atol=1e-3, rtol=1e-3)
    if !isnothing(pair)
        @test isapprox(∇1[4], ∇2[4]; atol=1e-3, rtol=1e-3)
    end
end

end
