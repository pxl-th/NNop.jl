struct FATileConfig{BM, BK, BN, TM, TN, rev_a, rev_b, rev_c} end

mma_non_acc_fn(c, r, x, y) = r
mma_acc_fn(c, r, x, y) = c + r

Base.@propagate_inbounds function mma!(
    c_shm::AbstractMatrix{T},
    a_shm::AbstractMatrix,
    b_shm::AbstractMatrix,
    cfg::Type{FATileConfig{BM, BK, BN, TM, TN, rev_a, rev_b, rev_c}},
    tidx,
    fn,
) where {T, BM, BK, BN, TM, TN, rev_a, rev_b, rev_c}
    thread_row = (tidx - 1) ÷ (BN ÷ TN)
    thread_col = (tidx - 1) % (BN ÷ TN)
    row_offset = thread_row * TM
    col_offset = thread_col * TN

    results = zeros(MMatrix{TM, TN, T})
    reg_m = MVector{TM, T}(undef)
    reg_n = MVector{TN, T}(undef)

    for dot_idx in 1:BK
        @unroll for reg_idx in 1:TM
            x, y = rev_a ? (dot_idx, row_offset + reg_idx) : (row_offset + reg_idx, dot_idx)
            reg_m[reg_idx] = a_shm[x, y]
        end
        @unroll for reg_idx in 1:TN
            x, y = rev_b ? (col_offset + reg_idx, dot_idx) : (dot_idx, col_offset + reg_idx)
            reg_n[reg_idx] = b_shm[x, y]
        end
        @unroll for res_idx_m in 1:TM
            @unroll for res_idx_n in 1:TN
                # TODO use fma
                results[res_idx_m, res_idx_n] += reg_m[res_idx_m] * reg_n[res_idx_n]
            end
        end
    end

    @unroll for res_idx_m in 1:TM
        @unroll for res_idx_n in 1:TN
            x, y = rev_c ?
                (col_offset + res_idx_n, row_offset + res_idx_m) :
                (row_offset + res_idx_m, col_offset + res_idx_n)
            c_shm[x, y] = fn(c_shm[x, y], results[res_idx_m, res_idx_n], x, y)
        end
    end
    return
end
