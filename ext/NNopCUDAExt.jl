module NNopCUDAExt

using CUDA
using NNop

function NNop._shared_memory(::CUDABackend, device_id::Integer)
    dev = collect(CUDA.devices())[device_id]
    return UInt64(CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK))
end

end
