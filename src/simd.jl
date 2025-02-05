@inline function vload(::Type{SIMD.Vec{N, T}}, ptr::Core.LLVMPtr{T, AS}) where {N, T, AS}
    alignment = sizeof(T) * N
    vec_ptr = Base.bitcast(Core.LLVMPtr{SIMD.Vec{N, T}, AS}, ptr)
    return unsafe_load(vec_ptr, 1, Val(alignment))
end

@inline function vstore!(ptr::Core.LLVMPtr{T, AS}, x::SIMD.Vec{N, T}) where {N, T, AS}
    alignment = sizeof(T) * N
    vec_ptr = Base.bitcast(Core.LLVMPtr{SIMD.Vec{N, T}, AS}, ptr)
    unsafe_store!(vec_ptr, x, 1, Val(alignment))
    return
end
