@testsetup module TSCore

export kab

if get(ENV, "NNOP_TEST_AMDGPU", "false") == "true"
    using AMDGPU
    kab = ROCBackend()
elseif get(ENV, "NNOP_TEST_CUDA", "false") == "true"
    using CUDA
    kab = CUDABackend()
else
    error("No GPU backend is set.")
end

end
