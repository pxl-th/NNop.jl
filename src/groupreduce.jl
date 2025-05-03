"""
    @groupreduce op val [groupsize]
Perform group reduction of `val` using `op`.
# Arguments
- `groupsize` specifies size of the workgroup.
    If a kernel does not specifies `groupsize` statically, then it is required to
    provide `groupsize`.
    Also can be used to perform reduction accross first `groupsize` threads
    (if `groupsize < @groupsize()`).
# Returns
Result of the reduction.
"""
macro groupreduce(op, val)
    :(__thread_groupreduce($(esc(:__ctx__)), $(esc(op)), $(esc(val)), Val(prod($(KA.groupsize)($(esc(:__ctx__)))))))
end
macro groupreduce(op, val, groupsize)
    :(__thread_groupreduce($(esc(:__ctx__)), $(esc(op)), $(esc(val)), Val($(esc(groupsize)))))
end

function __thread_groupreduce(__ctx__, op, val::T, ::Val{groupsize}) where {T, groupsize}
    storage = @localmem T groupsize

    lidx = @index(Local)
    @inbounds lidx ≤ groupsize && (storage[lidx] = val)
    @synchronize()

    s::UInt64 = groupsize ÷ 0x02
    while s > 0x00
        if (lidx - 0x01) < s
            other_idx = lidx + s
            if other_idx ≤ groupsize
                @inbounds storage[lidx] = op(storage[lidx], storage[other_idx])
            end
        end
        @synchronize()
        s >>= 0x01
    end

    if lidx == 0x01
        @inbounds val = storage[lidx]
    end
    return val
end
