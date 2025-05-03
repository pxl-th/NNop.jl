using NNop
using NNlib
using NNlib: ⊠, make_causal_mask, apply_attn_mask
using BenchmarkTools
using Zygote
using Random
using GPUArrays

import Adapt
import KernelAbstractions as KA

function naive_softmax(x; dims = 1)
    mx = maximum(x; dims)
    tmp = exp.(x .- mx)
    return tmp ./ sum(tmp; dims)
end

function naive_attention(q, k, v; causal::Bool)
    kt = permutedims(k, (2, 1, 3, 4))
    a = (kt ⊠ q) .* Float32(inv(sqrt(size(q, 1))))
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

function test_rms_norm(kab)
    emb, n = 1024, 1024
    x = Adapt.adapt(kab, rand(Float32, emb, n))
    w = Adapt.adapt(kab, rand(Float32, emb))

    y1 = naive_rms_norm(x, w; ϵ=1f-6)
    y2 = NNop.rms_norm(x, w; ϵ=1f-6)
    @assert y1 ≈ y2

    ∇n = Zygote.gradient(x, w) do x, w
        sum(naive_rms_norm(x, w; offset=1f0))
    end
    ∇f = Zygote.gradient(x, w) do x, w
        sum(NNop.rms_norm(x, w; offset=1f0))
    end
    @assert isapprox(∇n[1], ∇f[1]; atol=1f-6, rtol=1f-6)
    @assert isapprox(∇n[2], ∇f[2]; atol=1f-6, rtol=1f-6)

    cache = GPUArrays.AllocCache()

    println("Naїve RMS norm:")
    @btime GPUArrays.@cached $cache begin
        naive_rms_norm($x, $w)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Fused RMS norm:")
    @btime GPUArrays.@cached $cache begin
        NNop.rms_norm($x, $w)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Naїve RMS norm fwd + bwd:")
    @btime GPUArrays.@cached $cache begin
        Zygote.gradient($x, $w) do x, w
            sum(naive_rms_norm(x, w))
        end
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Fused RMS norm fwd + bwd:")
    @btime GPUArrays.@cached $cache begin
        Zygote.gradient($x, $w) do x, w
            sum(NNop.rms_norm(x, w))
        end
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)
    return
end

function test_softmax(kab)
    for seq_len in (32, 33, 63, 255, 256, 512, 1024, 2048)
        x = Adapt.adapt(kab, rand(Float32, seq_len, 4))
        y1 = naive_softmax(x; dims=1)
        y2 = NNop.online_softmax(x)
        @assert y1 ≈ y2

        ∇1 = Zygote.gradient(x) do x
            sum(naive_softmax(x))
        end
        ∇2 = Zygote.gradient(x) do x
            sum(online_softmax(x))
        end
        @assert isapprox(∇1[1], ∇2[1]; atol=1f-6, rtol=1f-6)
    end

    x = Adapt.adapt(kab, rand(Float32, 8192, 1024))
    cache = GPUArrays.AllocCache()

    y1 = naive_softmax(x; dims=1)
    y2 = NNop.online_softmax(x)
    @assert y1 ≈ y2

    println("Naїve softmax:")
    @btime GPUArrays.@cached $cache begin
        naive_softmax($x; dims=1)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Online softmax:")
    @btime GPUArrays.@cached $cache begin
        NNop.online_softmax($x)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    return
end

function test_flash_attention(kab)
    Random.seed!(0)
    T = Float32
    E, QL, KL, H, B = 64, 4096, 4096, 4, 4
    causal = false

    q = Adapt.adapt(kab, ones(T, E, QL, H, B))
    k = Adapt.adapt(kab, ones(T, E, KL, H, B))
    v = Adapt.adapt(kab, ones(T, E, KL, H, B))

    on = naive_attention(q, k, v; causal)
    o = NNop.flash_attention(q, k, v; causal)
    @assert on ≈ o

    ∇1 = Zygote.gradient(q, k, v) do q, k, v
        sum(naive_attention(q, k, v; causal))
    end
    ∇2 = Zygote.gradient(q, k, v) do q, k, v
        sum(NNop.flash_attention(q, k, v; causal))
    end

    @assert isapprox(∇1[1], ∇2[1]; atol=1e-3, rtol=1e-3)
    @assert isapprox(∇1[2], ∇2[2]; atol=1e-3, rtol=1e-3)
    @assert isapprox(∇1[3], ∇2[3]; atol=1e-3, rtol=1e-3)

    GC.gc(false)
    GC.gc(true)

    cache = GPUArrays.AllocCache()

    println("Naїve attention FWD:")
    @btime GPUArrays.@cached $cache begin
        naive_attention($q, $k, $v; causal=$causal)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Flash attention FWD:")
    @btime GPUArrays.@cached $cache begin
        NNop.flash_attention($q, $k, $v; causal=$causal)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Naїve attention FWD + BWD:")
    @btime GPUArrays.@cached $cache begin
        ∇ = Zygote.gradient($q, $k, $v) do q, k, v
            sum(naive_attention(q, k, v; causal=$causal))
        end
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Flash attention FWD + BWD:")
    @btime GPUArrays.@cached $cache begin
        Zygote.gradient($q, $k, $v) do q, k, v
            sum(NNop.flash_attention(q, k, v; causal=$causal))
        end
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    return
end
