
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
@generated function vresize(::Union{StaticInt{W},Val{W}}, v::Vec{L,T}) where {W,L,T}
    typ = LLVM_TYPES[T]
    mask = '<' * join(map(x->string("i32 ", x ≥ L ? L : x), 0:W-1), ", ") * '>'
    instrs = """
        %res = shufflevector <$L x $typ> %0, <$L x $typ> undef, <$W x i32> $mask
        ret <$W x $typ> %res
    """
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$W,$T}, Tuple{_Vec{$L,$T}}, data(v)))
    end
end
@generated function vresize(::Union{StaticInt{W},Val{W}}, v::T) where {W,T<:NativeTypes}
    typ = LLVM_TYPES[T]
    vtyp = vtype(W, typ)
    instrs = """
        %ie = insertelement $vtyp undef, $typ %0, i32 0
        ret $vtyp %ie
    """
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$W,$T}, Tuple{$T}, v))
    end
end

@generated function shufflevector(i::MM{W,X}, ::Val{I}) where {W,X,I}
    allincr = true
    L = length(I)
    for l ∈ 2:L
        allincr &= (I[l] == I[l-1] + 1)
    end
    allincr || return Expr(:block, Expr(:meta,:inline), :(shufflevector(Vec(i), Val{$I}())))
    Expr(:block, Expr(:meta,:inline), :(MM{$L,$X}( extractelement(i, $(first(I))) )))
end

@generated function vtranspose_a(xs::VecUnroll{2,8}, ::Val{3}, ::Val{8})
    quote
        $(Expr(:meta, :inline))
        x = unrolleddata(xs)
        x₀y₀z₀x₁y₁z₁x₂y₂ = x[1]
        z₂x₃y₃z₃x₄y₄z₄x₅ = x[2]
        y₅z₅x₆y₆z₆x₇y₇z₇ = x[3]
        
        x₀y₀z₀x₁x₄y₄z₄x₅ = shufflevector(x₀y₀z₀x₁y₁z₁x₂y₂, z₂x₃y₃z₃x₄y₄z₄x₅, Val{( 0, 1, 2, 3,12,13,14,15)}())
        y₁z₁x₂y₂y₅z₅x₆y₆ = shufflevector(x₀y₀z₀x₁y₁z₁x₂y₂, y₅z₅x₆y₆z₆x₇y₇z₇, Val{( 4, 5, 6, 7, 8, 9,10,11)}())
        z₂x₃y₃z₃z₆x₇y₇z₇ = shufflevector(z₂x₃y₃z₃x₄y₄z₄x₅, y₅z₅x₆y₆z₆x₇y₇z₇, Val{( 0, 1, 2, 3,12,13,14,15)}())
      
        y₀z₀y₁z₁y₄z₄y₅z₅ = shufflevector(x₀y₀z₀x₁x₄y₄z₄x₅, y₁z₁x₂y₂y₅z₅x₆y₆, Val{( 1, 2, 8, 9, 5, 6,12,13)}())
        x₂y₂x₃y₃x₆y₆x₇y₇ = shufflevector(y₁z₁x₂y₂y₅z₅x₆y₆, z₂x₃y₃z₃z₆x₇y₇z₇, Val{( 2, 3, 9,10, 6, 7,12,15)}())

        x₀x₁x₂x₃x₄x₅x₆x₇ = shufflevector(x₀y₀z₀x₁x₄y₄z₄x₅, x₂y₂x₃y₃x₆y₆x₇y₇, Val{( 0, 3, 8,10, 4, 7,12,14)}())
        y₀y₁y₂y₃y₄y₅y₆y₇ = shufflevector(y₀z₀y₁z₁y₄z₄y₅z₅, x₂y₂x₃y₃x₆y₆x₇y₇, Val{( 0, 2, 9,11, 4, 6,13,15)}())
        z₀z₁z₂z₃z₄z₅z₆z₇ = shufflevector(y₀z₀y₁z₁y₄z₄y₅z₅, z₂x₃y₃z₃z₆x₇y₇z₇, Val{( 1, 3, 8,11, 5, 7,12,15)}())

        return VecUnroll((x₀x₁x₂x₃x₄x₅x₆x₇, y₀y₁y₂y₃y₄y₅y₆y₇, z₀z₁z₂z₃z₄z₅z₆z₇))
    end
end

@generated function vtranspose_a(xs::VecUnroll{2,4}, ::Val{3}, ::Val{4})
    quote
        $(Expr(:meta, :inline))
        x = unrolleddata(xs)
        x0y0z0x1 = x[1]
        y1z1x2y2 = x[2]
        z2x3y3z3 = x[3]
        
        y0z0y1z1 = shufflevector(x0y0z0x1, y1z1x2y2, Val{(1,2,4,5)}())
        x2y2x3y3 = shufflevector(y1z1x2y2, z2x3y3z3, Val{(2,3,5,6)}())

        x0x1x2x3 = shufflevector(x0y0z0x1, x2y2x3y3, Val{(0,3,4,6)}())
        y0y1y2y3 = shufflevector(y0z0y1z1, x2y2x3y3, Val{(0,2,5,7)}())
        z0z1z2z3 = shufflevector(y0z0y1z1, z2x3y3z3, Val{(1,3,4,7)}())

        return VecUnroll((x0x1x2x3, y0y1y2y3, z0z1z2z3))
    end
end

@generated function vtranspose_b(xs::VecUnroll{2,4}, ::Val{3}, ::Val{4})
    quote
        $(Expr(:meta, :inline))
        x = unrolleddata(xs)
        v1_1 = x[1]
        v2_1 = x[2]
        v3_1 = x[3]
        
        v1_2 = shufflevector(v1_1, v2_1, Val{(0,3,6,2)}())
        v2_2 = shufflevector(v1_1, v2_1, Val{(1,4,7,5)}())
        v3_2 = v3_1

        v1_3 = shufflevector(v1_2, v3_2, Val{(0,1,2,5)}())
        v2_3 = v2_2
        v3_3 = shufflevector(v1_2, v3_2, Val{(3,6,4,7)}())

        v1_4 = v1_3
        v2_4 = shufflevector(v2_3, v3_3, Val{(0,1,2,5)}())
        v3_4 = shufflevector(v2_3, v3_3, Val{(4,3,6,7)}())

        return VecUnroll((v1_4, v2_4, v3_4))
    end
end

@generated function vtranspose(xs::VecUnroll{4,4}, ::Val{5}, ::Val{4})
    quote
        $(Expr(:meta, :inline))
        x = unrolleddata(xs)

        v1_1 = x[1]
        v2_1 = x[2]
        v3_1 = x[3]
        v4_1 = x[4]
        v5_1 = x[5]

        # step 1
        v1_2 = shufflevector(v1_1, v2_1, Val{(0, 5, 2, 7)}())
        v2_2 = shufflevector(v1_1, v2_1, Val{(1, 6, 3, 4)}())
        v3_2 = shufflevector(v3_1, v4_1, Val{(2, 3, 4, 7)}())
        v4_2 = shufflevector(v3_1, v4_1, Val{(0, 1, 5, 6)}())
        v5_2 = v5_1

        # step 2
        v1_3 = shufflevector(v1_2, v3_2, Val{(0, 1, 4, 7)}())
        v2_3 = v2_2
        v3_3 = shufflevector(v1_2, v3_2, Val{(2, 3, 6, 5)}())
        v4_3 = shufflevector(v4_2, v5_2, Val{(5, 0, 2, 6)}())
        v5_3 = shufflevector(v4_2, v5_2, Val{(4, 1, 3, 7)}())

        # step 3
        v1_4 = v1_3
        v2_4 = shufflevector(v2_3, v5_3, Val{(0, 1, 2, 4)}())
        v3_4 = shufflevector(v3_3, v4_3, Val{(0, 1, 2, 4)}())
        v4_4 = shufflevector(v3_3, v4_3, Val{(3, 5, 6, 7)}())
        v5_4 = shufflevector(v2_3, v5_3, Val{(3, 5, 6, 7)}())

        # step 4
        v1_5 = v1_4
        v2_5 = shufflevector(v2_4, v4_4, Val{(0, 1, 4, 3)}())
        v3_5 = v3_4
        v4_5 = shufflevector(v2_4, v4_4, Val{(2, 5, 6, 7)}())
        v5_5 = v5_4

        return VecUnroll((v1_5, v2_5, v3_5, v4_5, v5_5))
    end
end

@generated function vtranspose_simple(x::Vec, ::Val{M}, ::Val{N}) where {M,N}
    ordering = Tuple(transpose(reshape(0 : M * N - 1, M, N)))
    quote
        $(Expr(:meta, :inline))
        return VectorizationBase.shufflevector(x, Val{$ordering}())
    end
end

function bench_3x4_transpose_a!(A::Vector{Float64})
    pA = stridedpointer(A)

    i = 1
    while i < length(A)
        unroll = Unroll{1,1,3,1,4,zero(UInt)}((i,))
        v = vload(pA, unroll)
        vt = vtranspose_a(v, Val{3}(), Val{4}())
        vstore!(pA, vt, unroll)
        i += 12
    end

    return A
end

function bench_3x4_transpose_b!(A::Vector{Float64})
    pA = stridedpointer(A)

    i = 1
    while i < length(A)
        unroll = Unroll{1,1,3,1,4,zero(UInt)}((i,))
        v = vload(pA, unroll)
        vt = vtranspose_b(v, Val{3}(), Val{4}())
        vstore!(pA, vt, unroll)
        i += 12
    end

    return A
end
