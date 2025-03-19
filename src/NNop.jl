module NNop

using Random
using AMDGPU
using BFloat16s
using GPUArrays
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using BenchmarkTools
using NNlib: ⊠
using StaticArrays
using Zygote

import KernelAbstractions as KA
import SIMD

include("simd.jl")
# include("softmax.jl")
include("mma.jl")
include("attention.jl")
include("attention_bwd.jl")

function naive_softmax(x; dims = 1)
    mx = maximum(x; dims)
    tmp = exp.(x .- mx)
    return tmp ./ sum(tmp; dims)
end

function naive_attention(q, k, v)
    kt = permutedims(k, (2, 1, 3, 4))
    a = (kt ⊠ q) .* Float32(inv(sqrt(size(q, 1))))
    am = maximum(a; dims=1)
    return v ⊠ naive_softmax(a .- am; dims=1)
end

function test_softmax()
    for seq_len in (1024, 2048)
        x = ROCArray(rand(Float32, seq_len, 4))
        y1 = naive_softmax(x; dims=1)
        y2 = NNop.online_softmax(x)
        @assert y1 ≈ y2
    end

    x = ROCArray(rand(Float32, 8192, 1024))
    cache = GPUArrays.AllocCache()

    y1 = naive_softmax(x; dims=1)
    y2 = online_softmax(x)
    @assert y1 ≈ y2

    println("Naїve softmax:")
    @btime AMDGPU.@sync GPUArrays.@cached $cache naive_softmax($x; dims=1)
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Online softmax:")
    @btime AMDGPU.@sync GPUArrays.@cached $cache online_softmax($x)
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    return
end

function test_flash_attention()
    Random.seed!(0)
    T = Float32
    E, QL, KL, H, B = 64, 4096, 4096, 1, 1

    q = ROCArray(ones(T, E, QL, H, B))
    k = ROCArray(ones(T, E, KL, H, B))
    v = ROCArray(ones(T, E, KL, H, B))

    on = naive_attention(q, k, v)
    o, ms, ls = flash_attention(q, k, v)
    @assert on ≈ o

    ∇ = Zygote.gradient(q, k, v) do q, k, v
        sum(naive_attention(q, k, v))
    end

    Δ = ROCArray(ones(T, E, QL, H, B))

    dq, dk, dv = ∇flash_attention(Δ, o, ms, ls, q, k, v)
    @assert isapprox(dq, ∇[1]; atol=1e-3, rtol=1e-3)
    @assert isapprox(dk, ∇[2]; atol=1e-3, rtol=1e-3)
    @assert isapprox(dv, ∇[3]; atol=1e-3, rtol=1e-3)

    AMDGPU.synchronize()
    GC.gc(false)
    GC.gc(true)

    cache = GPUArrays.AllocCache()

    println("Naїve attention FWD:")
    @btime AMDGPU.@sync GPUArrays.@cached $cache naive_attention($q, $k, $v)
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Flash attention FWD:")
    @btime AMDGPU.@sync GPUArrays.@cached $cache flash_attention($q, $k, $v)
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Naїve attention FWD + BWD:")
    @btime AMDGPU.@sync GPUArrays.@cached $cache begin
        ∇ = Zygote.gradient($q, $k, $v) do q, k, v
            sum(naive_attention(q, k, v))
        end
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Flash attention FWD + BWD:")
    @btime AMDGPU.@sync GPUArrays.@cached $cache begin
        o, ms, ls = flash_attention($q, $k, $v)
        sum(o)
        dq, dk, dv = ∇flash_attention($Δ, o, ms, ls, $q, $k, $v)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    return
end

end
