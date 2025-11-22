@testitem "LayerNorm" setup=[TSCore] begin

import Adapt
import Zygote
using Statistics

function naive_layer_norm(x, w, b; ϵ::Float32 = 1f-6)
    μ = mean(x; dims=1)
    σ² = var(x; mean=μ, dims=1, corrected=false)
    (x .- μ) ./ sqrt.(σ² .+ ϵ) .* w .+ b
end

@testset "LayerNorm norm: emb=$emb, n=$n" for emb in (
    15, 255, 256, 257, 511, 512, 513, 1024,
), n in (
    1, 2, 4, 15, 16, 17, 23, 25,
)
    x = Adapt.adapt(kab, rand(Float32, emb, n))
    w = Adapt.adapt(kab, rand(Float32, emb))
    b = Adapt.adapt(kab, rand(Float32, emb))

    y1 = naive_layer_norm(x, w, b)
    y2 = NNop.layer_norm(x, w, b)
    @test y1 ≈ y2

    ∇n = Zygote.gradient(x, w, b) do x, w, b
        sum(naive_layer_norm(x, w, b))
    end
    ∇f = Zygote.gradient(x, w, b) do x, w, b
        sum(NNop.layer_norm(x, w, b))
    end
    @test isapprox(∇n[1], ∇f[1]; atol=1f-6, rtol=1f-6)
    @test isapprox(∇n[2], ∇f[2]; atol=1f-6, rtol=1f-6)
    @test isapprox(∇n[3], ∇f[3]; atol=1f-6, rtol=1f-6)
end

end
