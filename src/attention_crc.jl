within_gradient(x) = false
ChainRulesCore.rrule(::typeof(within_gradient), x) = true, _ -> (NoTangent(), NoTangent())

function flash_attention(q, k, v; causal::Bool)
    o = _flash_attention(q, k, v; causal)
    within_gradient(q) && return o
    @assert length(o) == 3
    return o[1]
end

function ChainRulesCore.rrule(::typeof(_flash_attention), q, k, v; causal::Bool)
    o, ms, ls = _flash_attention(q, k, v; causal)
    function _pullback(Δ)
        dq, dk, dv = ∇flash_attention(
            ChainRulesCore.unthunk(Δ),
            o, ms, ls, q, k, v; causal)
        return (
            ChainRulesCore.NoTangent(),
            dq, dk, dv)
    end
    return o, _pullback
end
