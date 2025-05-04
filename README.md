# NNop.jl

|GPU Backend|CI Status|
|-|-|
|[AMDGPU](https://github.com/JuliaGPU/AMDGPU.jl)|[![][buildkite-img-amdgpu]][buildkite-url]|
|[CUDA](https://github.com/JuliaGPU/CUDA.jl)|[![][buildkite-img-cuda]][buildkite-url]|

[buildkite-img]: https://badge.buildkite.com/b30cae2b9773cfd3464e6dad35de6a4a7151a6cb161da14c33.svg?branch=master
[buildkite-img-amdgpu]: https://badge.buildkite.com/b30cae2b9773cfd3464e6dad35de6a4a7151a6cb161da14c33.svg?branch=master&step=AMDGPU%20-%20Julia%201.11
[buildkite-img-cuda]: https://badge.buildkite.com/b30cae2b9773cfd3464e6dad35de6a4a7151a6cb161da14c33.svg?branch=master&step=CUDA%20-%20Julia%201.11
[buildkite-url]: https://buildkite.com/julialang/nnop-dot-jl

Pure Julia NN kernels:

- [Flash Attention](#flash-attention)
- [Fused Softmax](#fused-softmax)
- [Fused RMS Norm](#fused-rms-norm)
- [Fused Layer Norm](#fused-layer-norm)

## Flash Attention

Implementation of [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).

```julia
E, L, H, B = 64, 4096, 4, 4
causal = false

q = ROCArray(rand(Float32, E, L, H, B))
k = ROCArray(rand(Float32, E, L, H, B))
v = ROCArray(rand(Float32, E, L, H, B))

o = NNop.flash_attention(q, k, v; causal)
∇ = Zygote.gradient(q, k, v) do q, k, v
    sum(NNop.flash_attention(q, k, v; causal))
end
```

### Benchmarks:

For the problem size `(E=64, L=4096, H=4, B=4)`.

||Naїve attention|Flash Attention|
|-|-|-|
|FWD|||
|Execution time|60.987 ms|18.380 ms|
|Peak memory usage|5.044 GiB|16.500 MiB|
|FWD + BWD|||
|Execution time|1.154 s|306.960 ms|
|Peak memory usage|19.164 GiB|80.813 MiB|

### Features:

- Forward & backward passes.
- Arbitrary sequence length.
- FP32, FP16, BFP16 support.
- Variable sequence length.
- Causal masking.
- Zygote / ChainRules integration.

## Fused Softmax

Implementation of [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867).

```julia
x = ROCArray(rand(Float32, 8192, 1024))
y = NNop.online_softmax(x)
```

||Naїve Softmax|Online Softmax|
|-|-|-|
|Execution time|745.123 μs|61.600 μs|
|Peak memory usage|64.258 MiB|32.000 MiB|

## Fused RMS Norm

```julia
x = ROCArray(rand(Float32, 1024, 1024))
w = ROCArray(rand(Float32, 1024))
y = NNop.rms_norm(x, w)
∇ = Zygote.gradient(x, w) do x, w
    sum(NNop.rms_norm(x, w))
end
```

||Naїve RMS Norm|Fused RMS Norm|
|-|-|-|
|FWD|||
|Execution time|171.124 μs|48.432 μs|
|Peak memory usage|8.004 MiB|4.004 MiB|
|FWD + BWD|||
|Execution time|902.919 μs|241.838 μs|
|Peak memory usage|44.043 MiB|13.008 MiB|


## Fused Layer Norm

```julia
x = ROCArray(rand(Float32, 1024, 1024))
w = ROCArray(rand(Float32, 1024))
w = ROCArray(rand(Float32, 1024))
y = NNop.layer_norm(x, w)
∇ = Zygote.gradient(x, w, b) do x, w, b
    sum(NNop.layer_norm(x, w, b))
end
```

||Naїve Layer Norm|Fused Layer Norm|
|-|-|-|
|FWD|||
|Execution time|188.392 μs|48.175 μs|
|Peak memory usage|4.008 MiB|4.004 MiB|
|FWD + BWD|||
|Execution time|1.150 ms|293.969 μs|
|Peak memory usage|52.055 MiB|14.016 MiB|
