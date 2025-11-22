@testitem "Flash Attention" setup=[TSCore, TSAttention] begin

import Adapt
import Zygote

@testset "padmask=$use_padmask, pair=$use_pair, T=$T, E=$E, QL=$QL, KL=$KL" for use_padmask in (
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
    causal = false
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
