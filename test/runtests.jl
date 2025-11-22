using NNop
using NNlib: ⊠, make_causal_mask, apply_attn_mask
using Test
using Statistics

import Adapt
import Einops
import Zygote
import Pkg

#ENV["NNOP_TEST_AMDGPU"] = true
ENV["NNOP_TEST_CUDA"] = true

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

function build_doc_ids_cpu(lengths, L::Int, B::Int)
    ndocs = size(lengths, 1)
    doc_ids = Array{Int32}(undef, L, B)
    @inbounds for b in 1:B
        pos = 1
        for d in 1:ndocs
            len = Int(lengths[d, b])
            len == 0 && continue
            @assert len ≥ 0
            @assert pos + len - 1 ≤ L
            doc = Int32(d)
            doc_ids[pos:pos+len-1, b] .= doc
            pos += len
        end
        @assert pos - 1 == L
    end
    return doc_ids
end

function apply_lengths_mask(a, q_lengths, k_lengths)
    QL = size(a, 2)
    KL = size(a, 1)
    B  = size(a, 4)

    q_lengths_cpu = Array(q_lengths)
    k_lengths_cpu = Array(k_lengths)

    @assert size(q_lengths_cpu, 2) == B
    @assert size(k_lengths_cpu, 2) == B

    T = eltype(a)

    doc_mask_adapted = NNop.CRC.@ignore_derivatives begin
        q_doc_ids = build_doc_ids_cpu(q_lengths_cpu, QL, B)
        k_doc_ids = build_doc_ids_cpu(k_lengths_cpu, KL, B)

        doc_mask = Array{T}(undef, KL, QL, 1, B)
        @inbounds for b in 1:B, qpos in 1:QL, kpos in 1:KL
            qd = q_doc_ids[qpos, b]
            kd = k_doc_ids[kpos, b]
            same_doc = (qd != 0) && (qd == kd)
            doc_mask[kpos, qpos, 1, b] = same_doc ? zero(T) : typemin(T)
        end

        Adapt.adapt(kab, doc_mask)
    end

    return a .+ doc_mask_adapted
end

function naive_attention_impl(
    q, k, v, pair = nothing;
    causal::Bool,
)
    QH, KVH = size(q, 3), size(k, 3)
    if QH != KVH
        @assert QH % KVH == 0 "Number of query heads must be divisible by number of KV heads"
        num_q_per_kv = QH ÷ KVH
        k, v = repeat.((k, v), Einops.einops"d l h ... -> d l (num_q_per_kv h) ..."; num_q_per_kv)
    end
    kt = permutedims(k, (2, 1, 3, 4))
    a = (kt ⊠ q) .* inv(sqrt(size(q, 1)))
    if causal
        m = make_causal_mask(q)
        a = apply_attn_mask(a, m)
    end
    if !isnothing(pair)
        a = a .+ permutedims(pair, (3, 2, 1, 4)) #When it comes in as H-QL-KL-B
    end
    return v ⊠ naive_softmax(a; dims=1)
end

function naive_attention(
    q, k, v, pair = nothing;
    causal::Bool,
    lengths = nothing,
    q_lengths = nothing,
    k_lengths = nothing,
)
    if lengths !== nothing
        q_lengths === nothing || error("Specify either lengths or q_lengths, not both.")
        k_lengths === nothing || error("Specify either lengths or k_lengths, not both.")
        q_lengths = lengths
        k_lengths = lengths
    end

    if q_lengths === nothing && k_lengths === nothing
        return naive_attention_impl(q, k, v, pair; causal=causal)
    end

    !isnothing(pair) && error("pair is not supported together with lengths in naive_attention.")

    T = eltype(q)
    E, Lq, H, B = size(q)
    Lk = size(k, 2)

    q_lengths_cpu = Array(q_lengths)
    k_lengths_cpu = Array(k_lengths)

    ndocs_q = size(q_lengths_cpu, 1)
    ndocs_k = size(k_lengths_cpu, 1)
    @assert size(q_lengths_cpu, 2) == B
    @assert size(k_lengths_cpu, 2) == B

    o_batches = map(1:B) do b
        ndocs = max(ndocs_q, ndocs_k)
        q_lens = [d ≤ ndocs_q ? Int(q_lengths_cpu[d, b]) : 0 for d in 1:ndocs]
        k_lens = [d ≤ ndocs_k ? Int(k_lengths_cpu[d, b]) : 0 for d in 1:ndocs]

        if ndocs == 0
            zeros(T, E, 0, H, 1)
        else
            q_starts = cumsum(vcat(1, q_lens[1:end-1]))
            k_starts = cumsum(vcat(1, k_lens[1:end-1]))

            last_q_end = q_starts[end] + q_lens[end] - 1
            last_k_end = k_starts[end] + k_lens[end] - 1
            @assert last_q_end == Lq
            @assert last_k_end == Lk

            o_docs = [
                let q_len = q_lens[d], k_len = k_lens[d],
                    q_start = q_starts[d], k_start = k_starts[d]
                    if q_len == 0
                        zeros(T, E, 0, H, 1)
                    elseif k_len == 0
                        zeros(T, E, q_len, H, 1)
                    else
                        q_slice = q[:, q_start:(q_start + q_len - 1), :, b:b]
                        k_slice = k[:, k_start:(k_start + k_len - 1), :, b:b]
                        v_slice = v[:, k_start:(k_start + k_len - 1), :, b:b]

                        naive_attention_impl(q_slice, k_slice, v_slice, nothing; causal=causal)
                    end
                end
                for d in 1:ndocs
            ]

            cat(o_docs...; dims=2)
        end
    end

    return cat(o_batches...; dims=4)
end

function naive_rms_norm(x, w; offset::Float32 = 0f0, ϵ::Float32 = 1f-6)
    (w .+ offset) .* x ./ sqrt.(mean(x.^2; dims=1) .+ ϵ)
end

function naive_layer_norm(x, w, b; ϵ::Float32 = 1f-6)
    μ = mean(x; dims=1)
    σ² = var(x; mean=μ, dims=1, corrected=false)
    (x .- μ) ./ sqrt.(σ² .+ ϵ) .* w .+ b
end

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
    #=@testset "Online Softmax: T=$T, seq_len=$seq_len" for T in (
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
    end=#

    @testset "Flash Attention: causal=$causal, padmask=$use_padmask, pair=$use_pair, T=$T, E=$E, QL=$QL, KL=$KL" for causal in (
        false, true
    ), use_padmask in (
        false, true,
    ), use_pair in (
        false, true,
    ), T in (
        Float32, # TODO more types
    ), E in (
        16, 32, 64, # TODO test on higher if applicable
    ), QL in (
        255, 256, #511, 512, 1024,
    ), KL in (
        255, 256, #511, 512, 1024,
    )
        causal && QL != KL && continue

        H, B = 2, 3

        q = Adapt.adapt(kab, randn(T, E, QL, H, B))
        k = Adapt.adapt(kab, randn(T, E, KL, H, B))
        v = Adapt.adapt(kab, randn(T, E, KL, H, B))

        pair = nothing
        if use_pair
            pair = Adapt.adapt(kab, randn(T, H, QL, KL, B))
        end

        o1, ∇1 = Zygote.withgradient(q, k, v, pair) do q, k, v, pair
            sum(naive_attention(q, k, v, pair; causal))
        end
        o2, ∇2 = Zygote.withgradient(q, k, v, pair) do q, k, v, pair
            sum(NNop.flash_attention(q, k, v, pair; causal))
        end
        @test isapprox(o1, o2; atol=1e-3, rtol=1e-3)
        @test isapprox(∇1[1], ∇2[1]; atol=1e-3, rtol=1e-3)
        @test isapprox(∇1[2], ∇2[2]; atol=1e-3, rtol=1e-3)
        @test isapprox(∇1[3], ∇2[3]; atol=1e-3, rtol=1e-3)
        if !isnothing(pair)
            @test isapprox(∇1[4], ∇2[4]; atol=1e-3, rtol=1e-3)
        end
    end

    @testset "Grouped-Query Attention: QH=$QH, KVH=$KVH, causal=$causal, T=$T, E=$E, QL=$QL" for QH in (
        2, 4, 8,
    ), KVH in (
        1, 2,
    ), causal in (
        false, true,
    ), T in (
        Float32,
    ), E in (
        32, 64,
    ), QL in (
        256, 512,
    )
        QH % KVH == 0 || continue  # Skip invalid combinations
        QH == KVH && continue  # Skip regular MHA (already tested)
        
        KL = QL
        B = 2
        
        q = Adapt.adapt(kab, randn(T, E, QL, QH, B))
        k = Adapt.adapt(kab, randn(T, E, KL, KVH, B))
        v = Adapt.adapt(kab, randn(T, E, KL, KVH, B))
        
        o1, ∇1 = Zygote.withgradient(q, k, v) do q, k, v
            sum(naive_attention(q, k, v; causal))
        end
        o2, ∇2 = Zygote.withgradient(q, k, v) do q, k, v
            sum(NNop.flash_attention(q, k, v; causal))
        end
        @test isapprox(o1, o2; atol=1e-3, rtol=1e-3)
        @test isapprox(∇1[1], ∇2[1]; atol=1e-3, rtol=1e-3)
        @test isapprox(∇1[2], ∇2[2]; atol=1e-3, rtol=1e-3)
        @test isapprox(∇1[3], ∇2[3]; atol=1e-3, rtol=1e-3)
    end

    @testset "Flash Attention with lengths (self): causal=$causal, T=$T, E=$E, L=$L, QH=$QH, QH/KVH=$num_q_per_kv, ndocs=$ndocs" for causal in (
        false, true
    ), T in (
        Float32,
    ), E in (
        16,
    ), L in (
        32, 33,
    ), QH in (
        2, 4, 6
    ), num_q_per_kv in (
        2, 1
    ), ndocs in (
        2, 3,
    )
        KVH = QH ÷ num_q_per_kv
        B = 2
        lengths = zeros(Int, ndocs, B)
        @inbounds for b in 1:B
            remaining = L
            for d in 1:ndocs
                len = d == ndocs ? remaining : max(1, remaining ÷ (ndocs - d + 1))
                lengths[d, b] = len
                remaining -= len
            end
            @assert remaining == 0
        end

        q = Adapt.adapt(kab, randn(T, E, L, QH, B))
        k = Adapt.adapt(kab, randn(T, E, L, KVH, B))
        v = Adapt.adapt(kab, randn(T, E, L, KVH, B))

        o1, ∇1 = Zygote.withgradient(q, k, v) do q, k, v
            sum(naive_attention(q, k, v; causal=causal, lengths=lengths))
        end
        o2, ∇2 = Zygote.withgradient(q, k, v) do q, k, v
            sum(NNop.flash_attention(q, k, v; causal=causal, lengths=Adapt.adapt(kab, lengths)))
        end

        @test isapprox(o1, o2; atol=1e-3, rtol=1e-3)
        @test isapprox(∇1[1], ∇2[1]; atol=1e-3, rtol=1e-3)
        @test isapprox(∇1[2], ∇2[2]; atol=1e-3, rtol=1e-3)
        @test isapprox(∇1[3], ∇2[3]; atol=1e-3, rtol=1e-3)
    end

    @testset "Flash Attention with q_lengths/k_lengths (asym)" for T in (
        Float32,
    ), E in (
        16,
    )
        causal = false
        H, B = 2, 2
        Lq, Lk = 40, 48
        ndocs_q, ndocs_k = 3, 2

        q_lengths = zeros(Int, ndocs_q, B)
        k_lengths = zeros(Int, ndocs_k, B)

        @inbounds for b in 1:B
            remaining = Lq
            for d in 1:ndocs_q
                len = d == ndocs_q ? remaining : max(1, remaining ÷ (ndocs_q - d + 1))
                q_lengths[d, b] = len
                remaining -= len
            end
            @assert remaining == 0

            remaining = Lk
            for d in 1:ndocs_k
                len = d == ndocs_k ? remaining : max(1, remaining ÷ (ndocs_k - d + 1))
                k_lengths[d, b] = len
                remaining -= len
            end
            @assert remaining == 0
        end

        q = Adapt.adapt(kab, randn(T, E, Lq, H, B))
        k = Adapt.adapt(kab, randn(T, E, Lk, H, B))
        v = Adapt.adapt(kab, randn(T, E, Lk, H, B))

        o1, ∇1 = Zygote.withgradient(q, k, v) do q, k, v
            sum(naive_attention(q, k, v;
                causal=causal,
                q_lengths=q_lengths,
                k_lengths=k_lengths))
        end
        o2, ∇2 = Zygote.withgradient(q, k, v) do q, k, v
            sum(NNop.flash_attention(q, k, v;
                causal=causal,
                q_lengths=Adapt.adapt(kab, q_lengths),
                k_lengths=Adapt.adapt(kab, k_lengths)))
        end

        @test isapprox(o1, o2; atol=1e-3, rtol=1e-3)
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
