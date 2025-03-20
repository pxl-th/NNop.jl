using AMDGPU
using NNop
using NNlib: ⊠, make_causal_mask, apply_attn_mask
using Test

import Zygote

function naive_softmax(x; dims = 1)
    mx = maximum(x; dims)
    tmp = exp.(x .- mx)
    return tmp ./ sum(tmp; dims)
end

function naive_attention(q, k, v; causal::Bool)
    kt = permutedims(k, (2, 1, 3, 4))
    a = (kt ⊠ q) .* inv(sqrt(size(q, 1)))
    if causal
        m = make_causal_mask(q)
        a = apply_attn_mask(a, m)
    end
    am = maximum(a; dims=1)
    return v ⊠ naive_softmax(a .- am; dims=1)
end

@testset "NNop" begin
    # @testset "Online Softmax: seq_len=$seq_len" for seq_len in (1024, 2048)
    #     x = ROCArray(rand(Float32, seq_len, 4))
    #     y1 = naive_softmax(x; dims=1)
    #     y2 = NNop.online_softmax(x)
    #     @test y1 ≈ y2
    # end

    @testset "Flash Attention: causal=$causal, T=$T, E=$E, QL=$QL, KL=$KL" for causal in (
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

        q = ROCArray(rand(T, E, QL, H, B))
        k = ROCArray(rand(T, E, KL, H, B))
        v = ROCArray(rand(T, E, KL, H, B))

        on = naive_attention(q, k, v; causal)
        o, ms, ls = NNop.flash_attention(q, k, v; causal)
        @test isapprox(on, o; atol=1e-3, rtol=1e-3)

        ∇ = Zygote.gradient(q, k, v) do q, k, v
            sum(naive_attention(q, k, v; causal))
        end

        Δ = ROCArray(ones(T, E, QL, H, B))

        dq, dk, dv = NNop.∇flash_attention(Δ, o, ms, ls, q, k, v; causal)
        @test isapprox(dq, ∇[1]; atol=1e-3, rtol=1e-3)
        @test isapprox(dk, ∇[2]; atol=1e-3, rtol=1e-3)
        @test isapprox(dv, ∇[3]; atol=1e-3, rtol=1e-3)
    end
end
