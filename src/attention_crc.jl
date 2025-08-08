within_gradient(x) = false
CRC.rrule(::typeof(within_gradient), x) = true, _ -> (NoTangent(), NoTangent())

function flash_attention(
    q, k, v,
    pair::Maybe{AbstractArray{<:Real, 4}} = nothing;
    causal::Bool,
    kpad_mask::Maybe{AbstractMatrix{Bool}} = nothing,
)
    o = _flash_attention(q, k, v, pair; causal, kpad_mask)
    within_gradient(q) && return o
    @assert length(o) == 3
    return o[1]
end

function CRC.rrule(::typeof(_flash_attention),
    q, k, v,
    pair::Maybe{AbstractArray{<:Real, 4}} = nothing;
    causal::Bool,
    kpad_mask::Maybe{AbstractMatrix{Bool}} = nothing,
)
    o, ms, ls = _flash_attention(q, k, v, pair; causal, kpad_mask)

    function _pullback(Δ)
        dq, dk, dv, dpair = ∇flash_attention(
            CRC.unthunk(Δ),
            o, ms, ls, q, k, v, pair; causal, kpad_mask)
        return CRC.NoTangent(), dq, dk, dv, dpair
    end
    return o, _pullback
end

