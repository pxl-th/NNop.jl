using NNop
using NNlib: ⊠, make_causal_mask, apply_attn_mask
using Test
using Statistics

import Adapt
import Zygote
import Pkg

# ENV["NNOP_TEST_AMDGPU"] = true
# ENV["NNOP_TEST_CUDA"] = true

if get(ENV, "NNOP_TEST_AMDGPU", "false") == "true"
    Pkg.add("AMDGPU")
    using AMDGPU
    kab = ROCBackend()
elseif get(ENV, "NNOP_TEST_CUDA", "false") == "true"
    Pkg.add("CUDA")
    using CUDA
    kab = CUDABackend()
else
    error("No GPU backend is set.")
end

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

function naive_rms_norm(x, w; offset::Float32 = 0f0, ϵ::Float32 = 1f-6)
    (w .+ offset) .* x ./ sqrt.(mean(x.^2; dims=1) .+ ϵ)
end

function naive_layer_norm(x, w, b; ϵ::Float32 = 1f-6)
    μ = mean(x; dims=1)
    σ² = var(x; mean=μ, dims=1, corrected=false)
    (x .- μ) ./ sqrt.(σ² .+ ϵ) .* w .+ b
emd

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

@testset "NNop" begin
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

        q = Adapt.adapt(kab, rand(T, E, QL, H, B))
        k = Adapt.adapt(kab, rand(T, E, KL, H, B))
        v = Adapt.adapt(kab, rand(T, E, KL, H, B))

        on = naive_attention(q, k, v; causal)
        o = NNop.flash_attention(q, k, v; causal)
        @test isapprox(on, o; atol=1e-3, rtol=1e-3)

        ∇1 = Zygote.gradient(q, k, v) do q, k, v
            sum(naive_attention(q, k, v; causal))
        end
        ∇2 = Zygote.gradient(q, k, v) do q, k, v
            sum(NNop.flash_attention(q, k, v; causal))
        end

        @test isapprox(∇1[1], ∇2[1]; atol=1e-3, rtol=1e-3)
        @test isapprox(∇1[2], ∇2[2]; atol=1e-3, rtol=1e-3)
        @test isapprox(∇1[3], ∇2[3]; atol=1e-3, rtol=1e-3)
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
