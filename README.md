# NNop.jl

Pure Julia NN kernels.

> [!WARNING]
> The package is in the early stages and is not yet fully ready.

## Flash Attention

Implementation of [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).

```julia
E, L, H, B = 64, 4096, 4, 4
causal = false

q = ROCArray(rand(Float32, E, L, H, B))
k = ROCArray(rand(Float32, E, L, H, B))
v = ROCArray(rand(Float32, E, L, H, B))

o = flash_attention(q, k, v; causal)
∇ = Zygote.gradient(q, k, v) do q, k, v
    sum(flash_attention(q, k, v; causal))
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
- Arbitrary head sizes.
- FP32, FP16, BFP16 support.
- Variable sequence length.
- Causal masking.
- Zygote / ChainRules integration.

## Fused (online) Softmax

Implementation of [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867).

```julia
x = ROCArray(ones(Float32, 8192, 1024))
y = online_softmax(x)
```

||Naїve Softmax|Online Softmax|
|-|-|-|
|Execution time|745.123 μs|61.600 μs|
|Peak memory usage|64.258 MiB|32.000 MiB|
