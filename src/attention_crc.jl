within_gradient(x) = false
CRC.rrule(::typeof(within_gradient), x) = true, _ -> (NoTangent(), NoTangent())

function flash_attention(
    q, k, v,
    pair::Maybe{AbstractArray{<:Real, 4}} = nothing;
    causal::Bool,
    kpad_mask::Maybe{AbstractMatrix{Bool}} = nothing,
    lengths=nothing,
    q_lengths=nothing,
    k_lengths=nothing,
)
    if lengths !== nothing
        q_lengths === nothing || error("Specify either lengths or q_lengths, not both.")
        k_lengths === nothing || error("Specify either lengths or k_lengths, not both.")
        q_lengths = lengths
        k_lengths = lengths
    end

    o = _flash_attention(q, k, v, pair;
        causal,
        kpad_mask=kpad_mask,
        q_lengths=q_lengths,
        k_lengths=k_lengths)
    within_gradient(q) && return o
    @assert length(o) == 3
    return o[1]
end

function CRC.rrule(::typeof(_flash_attention),
    q, k, v,
    pair::Maybe{AbstractArray{<:Real, 4}} = nothing;
    causal::Bool,
    kpad_mask::Maybe{AbstractMatrix{Bool}} = nothing,
    lengths=nothing,
    q_lengths=nothing,
    k_lengths=nothing,
)
    if lengths !== nothing
        q_lengths === nothing || error("Specify either lengths or q_lengths, not both.")
        k_lengths === nothing || error("Specify either lengths or k_lengths, not both.")
        q_lengths = lengths
        k_lengths = lengths
    end

    o, ms, ls = _flash_attention(q, k, v, pair;
        causal,
        kpad_mask=kpad_mask,
        q_lengths=q_lengths,
        k_lengths=k_lengths)

    function _pullback(Δ)
        dq, dk, dv, dpair = ∇flash_attention(
            CRC.unthunk(Δ),
            o, ms, ls, q, k, v, pair;
            causal=causal,
            kpad_mask=kpad_mask,
            q_lengths=q_lengths,
            k_lengths=k_lengths)
        return CRC.NoTangent(), dq, dk, dv, dpair
    end
    return o, _pullback
end

