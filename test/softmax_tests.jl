@testitem "Softmax" setup=[TSCore] begin

import Adapt
import Zygote

function naive_softmax(x; dims = 1)
    mx = maximum(x; dims)
    tmp = exp.(x .- mx)
    return tmp ./ sum(tmp; dims)
end

@testset "Online Softmax: T=$T, seq_len=$seq_len" for T in (
    Float32, # TODO more types
), seq_len in (
    32, 33, 63, 255, 256, 511, 512, 513, 1024,
)
    x = Adapt.adapt(kab, rand(Float32, seq_len, 4))
    y1 = naive_softmax(x; dims=1)
    y2 = NNop.online_softmax(x)
    @test y1 ≈ y2

    ∇1 = Zygote.gradient(x) do x
        sum(naive_softmax(x))
    end
    ∇2 = Zygote.gradient(x) do x
        sum(NNop.online_softmax(x))
    end
    @assert isapprox(∇1[1], ∇2[1]; atol=1f-6, rtol=1f-6)
end

end
