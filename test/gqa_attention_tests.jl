@testitem "Flash Grouped-Query Attention" setup=[TSCore, TSAttention] begin

import Adapt
import Zygote

@testset "Grouped-Query Attention: QH=$QH, KVH=$KVH, causal=$causal, T=$T, E=$E, L=$L" for QH in (
    4, 6, 8,
), KVH in (
    1, 2,
), causal in (
    false, true,
), T in (
    Float32,
), E in (
    32, 64,
), L in (
    255, 256, 257, 512,
)
    B = 2
    q = Adapt.adapt(kab, randn(T, E, L, QH, B))
    k = Adapt.adapt(kab, randn(T, E, L, KVH, B))
    v = Adapt.adapt(kab, randn(T, E, L, KVH, B))

    o1, ∇1 = Zygote.withgradient(q, k, v) do q, k, v
        sum(naive_attention(q, k, v; causal, kpad_mask=nothing))
    end
    o2, ∇2 = Zygote.withgradient(q, k, v) do q, k, v
        sum(NNop.flash_attention(q, k, v; causal, kpad_mask=nothing))
    end
    @test isapprox(o1, o2; atol=1e-3, rtol=1e-3)
    @test isapprox(∇1[1], ∇2[1]; atol=1e-3, rtol=1e-3)
    @test isapprox(∇1[2], ∇2[2]; atol=1e-3, rtol=1e-3)
    @test isapprox(∇1[3], ∇2[3]; atol=1e-3, rtol=1e-3)
end

end
