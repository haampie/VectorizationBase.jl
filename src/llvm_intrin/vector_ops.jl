
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

@generated function vtranspose(xs::VecUnroll{L,W}) where {L,W}
    @assert L+1 == W

    i = 1

    ids = zeros(Int, W, 2)

    shuffles = Expr[]

    for j = 1:W
        v0 = Symbol(:v, j, :_0)
        push!(shuffles, :($v0 = vecs[$j]))
    end

    while 2^i ≤ W

        block_size = 2^i

        # Shuffle the indices
        copyto!(ids, 0:2W-1)
        for block = 1:block_size:W, row = block+block_size÷2:block+block_size-1
            ids[row, 1], ids[row-block_size÷2, 2] = ids[row-block_size÷2, 2], ids[row, 1]
        end

        # Generate shuffle statements
        for block = 1:block_size:W, col = block:block+block_size÷2-1
            from, to = col, col+block_size÷2

            v1_prev, v2_prev = Symbol(:v, from, :_, i - 1), Symbol(:v, to, :_, i - 1)
            v1_curr, v2_curr = Symbol(:v, from, :_, i), Symbol(:v, to, :_, i)

            push!(shuffles, :($v1_curr = shufflevector($v1_prev, $v2_prev, Val{tuple($(ids[:, 1]...))}())))
            push!(shuffles, :($v2_curr = shufflevector($v1_prev, $v2_prev, Val{tuple($(ids[:, 2]...))}())))
        end

        i += 1
    end

    final_vecs = [Symbol(:v, j, :_, i-1) for j=1:W]

    return quote
        vecs = unrolleddata(xs)
        $(shuffles...)
        VecUnroll(tuple($(final_vecs...)))
    end
end