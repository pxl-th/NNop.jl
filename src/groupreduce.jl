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

    local_idx = @index(Local)
    @inbounds local_idx ≤ groupsize && (storage[local_idx] = val)
    @synchronize()

    s::UInt64 = groupsize ÷ 0x02
    while s > 0x00
        if (local_idx - 0x01) < s
            other_idx = local_idx + s
            if other_idx ≤ groupsize
                @inbounds storage[local_idx] = op(storage[local_idx], storage[other_idx])
            end
        end
        @synchronize()
        s >>= 0x01
    end

    if local_idx == 0x01
        @inbounds val = storage[local_idx]
    end
    return val
end
