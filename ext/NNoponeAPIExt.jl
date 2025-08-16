module NNoponeAPIExt

using oneAPI
using NNop

function NNop._shared_memory(::oneAPIBackend, device_id::Integer)
    dev = oneAPI.devices()[device_id]
    return UInt64(oneAPI.compute_properties(dev).maxSharedLocalMemory)
end

end
