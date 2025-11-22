@testitem "RoPE" setup=[TSCore] begin

import Adapt
import Zygote

function rotate_half(x)
    half_dim = size(x, 1) ÷ 2
    x1 = x[1:half_dim, :, :, :]
    x2 = x[(half_dim + 1):end, :, :, :]
    return vcat(-x2, x1)
end

function naive_llama_rope(q, k; cos, sin)
    cos = reshape(cos, size(cos)[1:2]..., 1, size(cos, 3))
    sin = reshape(sin, size(sin)[1:2]..., 1, size(sin, 3))
    q = q .* cos .+ rotate_half(q) .* sin
    k = k .* cos .+ rotate_half(k) .* sin
    return q, k
end

@testset "Llama RoPE: L=$L, QH=$QH, KH=$KH" for L in (
    13, 255, 256, 257, 1024, 1025,
), QH in (
    1, 3, 4, 5,
), KH in (
    1, 3, 4, 5,
)
    dim = 16
    batch = 1
    emb = NNop.LlamaRotaryEmbedding(dim)

    position_ids = reshape(collect(0f0:Float32(L) - 1f0), :, 1)
    position_ids = repeat(position_ids; inner=(1, batch))

    cos, sin = emb(position_ids)
    cos = Adapt.adapt(kab, cos)
    sin = Adapt.adapt(kab, sin)
    q = Adapt.adapt(kab, ones(Float32, (dim, L, QH, batch)))
    k = Adapt.adapt(kab, ones(Float32, (dim, L, KH, batch)))

    q1, k1 = NNop.llama_rope(q, k; cos, sin)
    q2, k2 = naive_llama_rope(q, k; cos, sin)
    @test isapprox(q1, q2; atol=1f-6, rtol=1f-6)
    @test isapprox(k1, k2; atol=1f-6, rtol=1f-6)

    ∇1 = Zygote.gradient(q, k) do q, k
        qr, kr = NNop.llama_rope(q, k; cos, sin)
        sum(qr) + sum(kr)
    end
    ∇2 = Zygote.gradient(q, k) do q, k
        qr, kr = naive_llama_rope(q, k; cos, sin)
        sum(qr) + sum(kr)
    end
    @test isapprox(∇1[1], ∇2[1]; atol=1f-6, rtol=1f-6)
    @test isapprox(∇1[2], ∇2[2]; atol=1f-6, rtol=1f-6)
end

end
