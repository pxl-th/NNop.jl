module NNop

using Random
using BFloat16s
using GPUArrays
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using BenchmarkTools
using NNlib: ⊠, make_causal_mask, apply_attn_mask
using StaticArrays
using Zygote

import Adapt
import KernelAbstractions as KA
import SIMD

include("simd.jl")
include("groupreduce.jl")
include("softmax.jl")
include("mma.jl")
include("attention.jl")
include("attention_bwd.jl")

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

function test_softmax(kab)
    for seq_len in (1024, 2048)
        x = Adapt.adapt(kab, rand(Float32, seq_len, 4))
        y1 = naive_softmax(x; dims=1)
        y2 = NNop.online_softmax(x)
        @assert y1 ≈ y2
    end

    x = Adapt.adapt(kab, rand(Float32, 8192, 1024))
    cache = GPUArrays.AllocCache()

    y1 = naive_softmax(x; dims=1)
    y2 = online_softmax(x)
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
        online_softmax($x)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    return
end

function test_flash_attention(kab)
    Random.seed!(0)
    T = Float32
    E, QL, KL, H, B = 64, 64, 64, 1, 1
    causal = false

    q = Adapt.adapt(kab, ones(T, E, QL, H, B))
    k = Adapt.adapt(kab, ones(T, E, KL, H, B))
    v = Adapt.adapt(kab, ones(T, E, KL, H, B))

    on = naive_attention(q, k, v; causal)
    o, ms, ls = flash_attention(q, k, v; causal)
    @assert on ≈ o

    ∇ = Zygote.gradient(q, k, v) do q, k, v
        sum(naive_attention(q, k, v; causal))
    end

    Δ = Adapt.adapt(kab, ones(T, E, QL, H, B))

    dq, dk, dv = ∇flash_attention(Δ, o, ms, ls, q, k, v; causal)
    @assert isapprox(dq, ∇[1]; atol=1e-3, rtol=1e-3)
    @assert isapprox(dk, ∇[2]; atol=1e-3, rtol=1e-3)
    @assert isapprox(dv, ∇[3]; atol=1e-3, rtol=1e-3)

    GC.gc(false)
    GC.gc(true)
    return

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
        flash_attention($q, $k, $v; causal=$causal)
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
        o, ms, ls = flash_attention($q, $k, $v; causal=$causal)
        sum(o)
        dq, dk, dv = ∇flash_attention($Δ, o, ms, ls, $q, $k, $v; causal=$causal)
        KA.synchronize($kab)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    return
end

end
