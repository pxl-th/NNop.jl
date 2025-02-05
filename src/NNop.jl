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
include("softmax.jl")
include("attention.jl")
include("attention_v2.jl")
include("attention_bwd_v2.jl")

function test_softmax()
    x = ROCArray(ones(Float32, 8192, 1024))
    cache = GPUArrays.AllocCache()

    y1 = naive_softmax(x; dims=1)
    y2 = online_softmax(x)
    y3 = online_softmax_simd(x)
    @assert y1 ≈ y2
    @assert y1 ≈ y3

    println("Naїve softmax:")
    @btime AMDGPU.@sync GPUArrays.@cached $cache naive_softmax($x; dims=1)
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Online softmax:")
    @btime AMDGPU.@sync GPUArrays.@cached $cache online_softmax($x)
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    println("Online softmax SIMD:")
    @btime AMDGPU.@sync GPUArrays.@cached $cache online_softmax_simd($x)
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    return
end

function test_flash_attention()
    Random.seed!(0)
    E = 64
    L = 4096
    H, B = 4, 4
    T = Float32

    q = ROCArray(rand(T, E, L, H, B))
    k = ROCArray(rand(T, E, L, H, B))
    v = ROCArray(rand(T, E, L, H, B))

    on = naive_attention(q, k, v)

    o, ms, ls = flash_attention(q, k, v)
    @assert on ≈ o

    o2, ms2, ls2 = flash_attention_v2(q, k, v)
    @assert on ≈ o2

    ∇ = Zygote.gradient(q, k, v) do q, k, v
        sum(naive_attention(q, k, v))
    end

    Δ = ROCArray(ones(T, E, L, H, B))

    dq, dk, dv = ∇flash_attention(Δ, o, ms, ls, q, k, v)
    @assert isapprox(dq, ∇[1]; atol=1e-3, rtol=1e-3)
    @assert dk ≈ ∇[2]
    @assert dv ≈ ∇[3]

    dq2, dk2, dv2 = ∇flash_attention_v2(Δ, o2, ms2, ls2, q, k, v)
    @assert isapprox(dq2, ∇[1]; atol=1e-2, rtol=1e-2)
    @assert isapprox(dk2, ∇[2]; atol=1e-3, rtol=1e-3)
    @assert isapprox(dv2, ∇[3]; atol=1e-3, rtol=1e-3)

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

    println("Flash attention V2 FWD:")
    @btime AMDGPU.@sync GPUArrays.@cached $cache flash_attention_v2($q, $k, $v)
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

    println("Flash attention V2 FWD + BWD:")
    @btime AMDGPU.@sync GPUArrays.@cached $cache begin
        o, ms, ls = flash_attention_v2($q, $k, $v)
        sum(o)
        dq, dk, dv = ∇flash_attention_v2($Δ, o, ms, ls, $q, $k, $v)
    end
    println(" - Peak memory usage: $(Base.format_bytes(sizeof(cache)))")
    GPUArrays.unsafe_free!(cache)

    return
end

end
