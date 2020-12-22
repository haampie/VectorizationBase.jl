
function shufflevector_instrs(W, T, I, two_operands)
    typ = LLVM_TYPES[T]
    vtyp1 = "<$W x $typ>"
    M = length(I)
    vtyp3 = "<$M x i32>"
    vtypr = "<$M x $typ>"
    mask = '<' * join(map(x->string("i32 ", x), I), ", ") * '>'
    v2 = two_operands ? "%1" : "undef"
    M, """
        %res = shufflevector $vtyp1 %0, $vtyp1 $v2, $vtyp3 $mask
        ret $vtypr %res
    """
end
@generated function shufflevector(v1::Vec{W,T}, v2::Vec{W,T}, ::Val{I}) where {W,T,I}
    M, instrs = shufflevector_instrs(W, T, I, true)
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$M,$T}, Tuple{_Vec{$W,$T}, _Vec{$W,$T}}, data(v1), data(v2)))
    end
end
@generated function shufflevector(v1::Vec{W,T}, ::Val{I}) where {W,T,I}
    M, instrs = shufflevector_instrs(W, T, I, false)
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$M,$T}, Tuple{_Vec{$W,$T}}, data(v1)))
    end
end

@inline function vtranspose(xs::VecUnroll{3,8}, ::Val{4})
    vecs = unrolleddata(xs)

    v1_1 = vecs[1]
    v2_1 = vecs[2]
    v3_1 = vecs[3]
    v4_1 = vecs[4]

    # skip the first iteration. second iteration acts on distance 2.
    v1_2 = shufflevector(v1_1, v2_1, Val{(0,  4,  8, 12,  1,  5,  9, 13)}())
    v2_2 = shufflevector(v1_1, v2_1, Val{(2,  6, 10, 14,  3,  7, 11, 15)}())
    v3_2 = shufflevector(v3_1, v4_1, Val{(0,  4,  8, 12,  1,  5,  9, 13)}())
    v4_2 = shufflevector(v3_1, v4_1, Val{(2,  6, 10, 14,  3,  7, 11, 15)}())

    v1_3 = shufflevector(v1_2, v3_2, Val{(0,  1,  2,  3,  8,  9, 10, 11)}())
    v2_3 = shufflevector(v1_2, v3_2, Val{(4,  5,  6,  7, 12, 13, 14, 15)}())
    v3_3 = shufflevector(v2_2, v4_2, Val{(0,  1,  2,  3,  8,  9, 10, 11)}())
    v4_3 = shufflevector(v2_2, v4_2, Val{(4,  5,  6,  7, 12, 13, 14, 15)}())

    return VecUnroll((v1_3, v2_3, v3_3, v4_3))
end

@inline function vtranspose(xs::VecUnroll{2,8}, ::Val{3})
    vecs = unrolleddata(xs)

    v1_1 = vecs[1]
    v2_1 = vecs[2]
    v3_1 = vecs[3]

    v1_2 = shufflevector(v1_1, v2_1, Val{(0,  1,  3,  4,  6,  9, 12, 15)}())
    v2_2 = shufflevector(v1_1, v2_1, Val{(2,  5,  7,  8, 10, 11, 13, 14)}())
    v3_2 = v3_1

    v1_3 = shufflevector(v1_2, v3_2, Val{(0,  2,  4,  5,  6,  7, 10, 13)}())
    v2_3 = shufflevector(v2_2, v3_2, Val{(2,  4,  6,  8, 10, 11, 13, 14)}())
    v3_3 = shufflevector(v2_2, v3_2, Val{(0,  1,  3,  5,  7,  9, 12, 15)}())

    v1_4 = v1_3
    v2_4 = shufflevector(v1_2, v2_3, Val{(1,  3,  8,  9, 10, 11, 13, 14)}())
    v3_4 = v3_3

    return VecUnroll((v1_4, v2_4, v3_4))
end

@inline function vtranspose(xs::VecUnroll{1,8}, ::Val{32})
    vecs = unrolleddata(xs)

    v1_1 = vecs[1]
    v2_1 = vecs[2]

    v1_2 = shufflevector(v1_1, v2_1, Val{(0,  2,  4,  6,  8, 10, 12, 14)}())
    v2_2 = shufflevector(v1_1, v2_1, Val{(1,  3,  5,  7,  9, 11, 13, 15)}())\

    return VecUnroll((v1_2, v2_2))
end

@generated function vtranspose(xs::VecUnroll{L,W,T,V}, ::Val{R}) where {L,W,T,V,R}
    # the number of Vecs is independent of the number of rows.
    nvecs = L + 1

    rows = R

    # Only assume we have a rectangular matrix.
    @assert (nvecs * W) % rows == 0

    cols = nvecs * W ÷ rows
    shuffles = Expr[]

    # Number of Vec's per row.
    block_rows = ÷(rows, W, RoundUp)
    block_cols = ÷(cols, W, RoundUp)

    vec_idx, vec_offset = 1, 0

    # Example with vector size == 2, and rows == 3, 4 input vectors
    # then we pad them into size 4x4:
    #   j=1   j=2
    #     +-----+----+
    # i=1 | a b | c .| and store them in 8 registers indexed
    #     | a b | d .| by block (i, j) and column k as
    #     | ----+----+ v_i_j_k_0 where 0 means 0'th iteration
    # i=2 | b c | d .| of the shuffle algorithm.
    #     | . . | . .|
    #     +-----+----+
    # and apply the tranpose kernel on every 2x2 block.
    # lastly we tranpose the blocks.

    for bj = 1:W:cols # loop over blocks →, active col idx = bj+cj
        for cj = 0:W-1 # loop over cols per block →
            for bi = 1:W:rows # loop over blocks ↓

                vec_name = Symbol(:v, :_, bi, :_, bj, :_, cj+1, :_0)

                # pad columns with zeros..
                if vec_idx > nvecs
                    push!(shuffles, :($vec_name = zero($V)))
                else
                    consume = min(W, rows - bi + 1)

                    # If we just take the first `consume` elements of
                    # a vectors, you can just as well take the full vector.
                    if vec_offset == 0
                        push!(shuffles, :($vec_name = vecs[$vec_idx]))
                    else                    
                        if vec_idx < nvecs
                            # If we have a next vector, consume from that.
                            ids = circshift(0:2W-1, -vec_offset)[1:W]
                            perm = :(Val{(tuple($(ids...)))}())
                            push!(shuffles, :($vec_name = shufflevector(vecs[$vec_idx], vecs[$(vec_idx+1)], $perm)))
                        else
                            ids = circshift(0:W-1, -vec_offset)[1:W]
                            perm = :(Val{(tuple($(ids...)))}())
                            push!(shuffles, :($vec_name = shufflevector(vecs[$vec_idx], $perm)))
                        end
                    end

                    vec_offset += consume
                    if vec_offset ≥ W
                        vec_offset -= W
                        vec_idx += 1
                    end
                end
            end
        end
    end

    iter = 1

    # now shuffle within each block of size W × W using the W lg W algorithm.
    while 2^iter ≤ W
        block_size = 2^iter

        for bj = 1:W:cols # loop over blocks →, active col idx = bj+cj
            for bi = 1:W:rows # loop over blocks ↓
                ids = zeros(Int, W, 2)

                # Shuffle the indices
                copyto!(ids, 0:2W-1)
                for block = 1:block_size:W, row = block+block_size÷2:block+block_size-1
                    ids[row, 1], ids[row-block_size÷2, 2] = ids[row-block_size÷2, 2], ids[row, 1]
                end

                # Generate shuffle statements
                for block = 1:block_size:W, col = block:block+block_size÷2-1
                    from, to = col, col+block_size÷2

                    v1_prev, v2_prev = Symbol(:v_, bi, :_, bj, :_, from, :_, iter - 1), Symbol(:v_, bi, :_, bj, :_, to, :_, iter - 1)
                    v1_curr, v2_curr = Symbol(:v_, bi, :_, bj, :_, from, :_, iter), Symbol(:v_, bi, :_, bj, :_, to, :_, iter)

                    push!(shuffles, :($v1_curr = shufflevector($v1_prev, $v2_prev, Val{tuple($(ids[:, 1]...))}())))
                    push!(shuffles, :($v2_curr = shufflevector($v1_prev, $v2_prev, Val{tuple($(ids[:, 2]...))}())))
                end
            end
        end
        iter += 1
    end

    # finally return the vecs we need.
    vecs = []

    for bj = 1:W:rows, cj = 0:W-1, bi = 1:W:cols
        bj + cj > rows && continue
        push!(vecs, Symbol(:v_, bj, :_, bi, :_, cj + 1, :_, iter - 1))
    end

    return quote
        $(Expr(:meta,:inline))
        vecs = unrolleddata(xs)
        $(shuffles...)
        return VecUnroll(tuple($(vecs...)))
    end
end