module NNop

using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using StaticArrays
using Memoize: @memoize
using LRUCache: LRU

import ChainRulesCore
import KernelAbstractions as KA
import SIMD

include("simd.jl")
include("groupreduce.jl")
include("softmax.jl")
include("mma.jl")
include("attention.jl")
include("attention_bwd.jl")
include("attention_crc.jl")
include("rms_norm.jl")

@memoize LRU{Tuple{Any, Integer}, UInt64}(maxsize=32) shared_memory(kab, device_id::Integer) =
    _shared_memory(kab, device_id)

_shared_memory(kab, device_id::Integer) = error("Not implemented.")

end
