@testitem "RMS Norm" setup=[TSCore] begin

import Adapt
import Zygote
using Statistics

function naive_rms_norm(x, w; offset::Float32 = 0f0, ϵ::Float32 = 1f-6)
    return (w .+ offset) .* x ./ sqrt.(mean(x.^2; dims=1) .+ ϵ)
end

@testset "RMS norm: emb=$emb, n=$n, offset=$offset" for emb in (
    15, 255, 256, 257, 511, 512, 513, 1024,
), n in (
    1, 2, 4, 15, 16, 17, 23, 25,
), offset in (
    0f0, 1f0,
)
    x = Adapt.adapt(kab, rand(Float32, emb, n))
    w = Adapt.adapt(kab, rand(Float32, emb))

    y1 = naive_rms_norm(x, w; offset)
    y2 = NNop.rms_norm(x, w; offset)
    @test y1 ≈ y2

    ∇n = Zygote.gradient(x, w) do x, w
        sum(naive_rms_norm(x, w; offset))
    end
    ∇f = Zygote.gradient(x, w) do x, w
        sum(NNop.rms_norm(x, w; offset))
    end
    @test isapprox(∇n[1], ∇f[1]; atol=1f-6, rtol=1f-6)
    @test isapprox(∇n[2], ∇f[2]; atol=1f-6, rtol=1f-6)
end

end
