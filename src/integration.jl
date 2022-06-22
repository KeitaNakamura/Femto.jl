abstract type IntegrationStyle end

function integrate(f, fieldtype::FieldType, element::Element)
    integrate(f, TensorStyle(f, element), fieldtype, element)
end

function integrate(f, style::IntegrationStyle, fieldtype::FieldType, element::Element)
    sum(1:num_quadpoints(element)) do qp
        @_inline_meta
        @inbounds build_element(f, style, fieldtype, element, qp)
    end
end

###############
# TensorStyle #
###############

struct TensorStyle{vector_or_matrix} <: IntegrationStyle end

## constructors
TensorStyle(f, element::Element) = TensorStyle(f, typeof(element))
@pure function TensorStyle(f, ::Type{<: BodyElement})
    nargs = last(methods(f)).nargs - 1
    nargs == 2 && return TensorStyle{:vector}()
    nargs == 3 && return TensorStyle{:matrix}()
    error("wrong number of arguments in `integrate`, use `(index, u, v)` for matrix or `(index, v)` for vector")
end
@pure function TensorStyle(f, ::Type{<: FaceElement})
    nargs = last(methods(f)).nargs - 1
    nargs == 3 && return TensorStyle{:vector}()
    error("wrong number of arguments in `integrate`, use `(index, v, normal)`")
end

## shape values and gradients
# ScalarField
shape_values(::TensorStyle, ::ScalarField, element::Element, qp::Int) = (@_propagate_inbounds_meta; element.N[qp])
shape_gradients(::TensorStyle, ::ScalarField, element::Element, qp::Int) = (@_propagate_inbounds_meta; element.dNdx[qp])
# VectorField
@generated function shape_values(::TensorStyle, ::VectorField, element::Element{T, dim, <: Any, L}, qp::Int) where {T, dim, L}
    exps = Expr[]
    for k in 1:L, d in 1:dim
        vals = [ifelse(i==d, :(N[$k]), :(zero($T))) for i in 1:dim]
        vec = :(Vec{dim, T}($(vals...)))
        push!(exps, vec)
    end
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        N = element.N[qp]
        SVector($(exps...))
    end
end
@generated function shape_gradients(::TensorStyle, ::VectorField, element::Element{T, dim, <: Any, L}, qp::Int) where {T, dim, L}
    exps = Expr[]
    for k in 1:L, d in 1:dim
        grads = [ifelse(i==d, :(dNdx[$k][$j]), :(zero($T))) for i in 1:dim, j in 1:dim]
        mat = :(Mat{dim, dim, T}($(grads...)))
        push!(exps, mat)
    end
    quote
        @_inline_meta
        @_propagate_inbounds_meta
        dNdx = element.dNdx[qp]
        SVector($(exps...))
    end
end

## build element matrix and vector
# BodyElement
@inline function build_element(f, style::TensorStyle{vector_or_matrix}, fieldtype::FieldType, element::Element, qp::Int) where {vector_or_matrix}
    @_propagate_inbounds_meta
    N = shape_values(style, fieldtype, element, qp)
    dNdx = shape_gradients(style, fieldtype, element, qp)
    detJdΩ = element.detJdΩ[qp]
    u = v = map(dual, N, dNdx)
    vector_or_matrix == :vector && return build_element_vector(f, qp, v)    * detJdΩ
    vector_or_matrix == :matrix && return build_element_matrix(f, qp, v, u) * detJdΩ
    error("unreachable")
end
# FaceElement
@inline function build_element(f, style::TensorStyle{:vector}, fieldtype::FieldType, element::FaceElement, qp::Int)
    @_propagate_inbounds_meta
    v = shape_values(style, fieldtype, element, qp)
    detJdΩ = element.detJdΩ[qp]
    normal = element.normal[qp]
    build_element_vector(f, qp, v, normal) * detJdΩ
end

# helpers
@generated function build_element_vector(f, qp, N::SVector{L}, args...) where {L}
    exps = [:(f(qp, N[$i], args...)) for i in 1:L]
    quote
        @_inline_meta
        @inbounds SVector{$L}($(exps...))
    end
end
@generated function build_element_matrix(f, qp, v::SVector{L1}, u::SVector{L2}) where {L1, L2}
    exps = [:(f(qp, u[$j], v[$i])) for i in 1:L1, j in 1:L2]
    quote
        @_inline_meta
        @inbounds SMatrix{$L1, $L2}($(exps...))
    end
end


###############
# MatrixStyle #
###############

struct MatrixStyle{vector_or_matrix} <: IntegrationStyle end

# constructors
MatrixStyle(f, element::Element) = MatrixStyle(f, typeof(element))
@pure function MatrixStyle(f, ::Type{<: BodyElement})
    nargs = last(methods(f)).nargs - 1
    nargs == 2 && return MatrixStyle{:vector}()
    nargs == 3 && return MatrixStyle{:matrix}()
    error("wrong number of arguments in `integrate`, use `(index, Nu, Nv)` for matrix or `(index, Nv)` for vector")
end
@pure function MatrixStyle(f, ::Type{<: FaceElement})
    nargs = last(methods(f)).nargs - 1
    nargs == 3 && return MatrixStyle{:vector}()
    error("wrong number of arguments in `integrate`, use `(index, Nv, normal)`")
end

## ShapeGradientMatrix
# for `symmetric(∇(N))`
struct ShapeGradientMatrix{m, n, T, L} <: StaticMatrix{m, n, T}
    data::NTuple{L, T}
end
@inline Base.getindex(dNdx::ShapeGradientMatrix, i::Int) = (@_propagate_inbounds_meta; dNdx.data[i])

# B-matrix
Bmatrix(dNdx::Vec{1}) = @SMatrix[dNdx[1]]
Bmatrix(dNdx::Vec{2}) = @SMatrix[dNdx[1] 0
                                 0       dNdx[2]
                                 dNdx[2] dNdx[1]]
Bmatrix(dNdx::Vec{3}) = @SMatrix[dNdx[1] 0       0
                                 0       dNdx[2] 0
                                 0       0       dNdx[3]
                                 dNdx[2] dNdx[1] 0
                                 dNdx[3] 0       dNdx[1]
                                 0       dNdx[3] dNdx[2]]
# create B-matrix from gradient matrix
function Tensorial.symmetric(dNdx::ShapeGradientMatrix{dim, n}) where {dim, n}
    reduce(hcat, ntuple(Val(n÷dim)) do J
        j = dim*(J-1) + 1
        Bmatrix(Vec{dim}(i -> dNdx[i, j+i-1]))
    end)
end

## ShapeValueMatrix
# for `∇(N)`
struct ShapeValueMatrix{m, n, T, L, A} <: StaticMatrix{m, n, T}
    data::NTuple{L, T}
    dNdx::A
end
function ShapeValueMatrix(N::SMatrix{m, n, T, L}, dNdx::A) where {m, n, T, L, A}
    ShapeValueMatrix{m, n, T, L, A}(Tuple(N), dNdx)
end
function ShapeValueMatrix(N::Adjoint{T, SVector{L, T}}, dNdx::A) where {T, L, A}
    ShapeValueMatrix{1, L, T, L, A}(Tuple(N), dNdx)
end
@inline Base.getindex(N::ShapeValueMatrix, i::Int) = (@_propagate_inbounds_meta; N.data[i])
∇(N::ShapeValueMatrix) = N.dNdx
 # Nv' becomes L×1 vector without this `adjoint`
LinearAlgebra.adjoint(N::ShapeValueMatrix{1}) = SVector(N.data)

## shape values and gradients
# ScalarField
shape_values(::MatrixStyle, ::ScalarField, element::Element, qp::Int) = (@_propagate_inbounds_meta; element.N[qp]')
function shape_gradients(::MatrixStyle, ::ScalarField, element::Element{T, dim}, qp::Int) where {T, dim}
    @_propagate_inbounds_meta
    mapreduce(SVector{dim, T}, hcat, element.dNdx[qp])
end
# VectorField
function shape_values(::MatrixStyle, ::VectorField, element::Element{T, dim}, qp::Int) where {T, dim}
    @_propagate_inbounds_meta
    mapreduce(Nᵢ -> diagm(Nᵢ*ones(SVector{dim, T})), hcat, element.N[qp])
end
function shape_gradients(::MatrixStyle, ::VectorField, element::Element{T, dim}, qp::Int) where {T, dim}
    @_propagate_inbounds_meta
    dNdx = mapreduce(diagm, hcat, element.dNdx[qp])
    m, n = size(dNdx)
    ShapeGradientMatrix{m, n, T, m*n}(Tuple(dNdx))
end

@inline function build_element(f, style::MatrixStyle{vector_or_matrix}, fieldtype::FieldType, element::BodyElement, qp::Int) where {vector_or_matrix}
    @_propagate_inbounds_meta
    Nu = Nv = ShapeValueMatrix(
        shape_values(style, fieldtype, element, qp),
        shape_gradients(style, fieldtype, element, qp),
    )
    detJdΩ = element.detJdΩ[qp]
    vector_or_matrix == :vector && return f(qp, Nv)     * detJdΩ
    vector_or_matrix == :matrix && return f(qp, Nu, Nv) * detJdΩ
    error("unreachable")
end

@inline function build_element(f, style::MatrixStyle{:vector}, fieldtype::FieldType, element::FaceElement, qp::Int)
    @_propagate_inbounds_meta
    Nv = shape_values(style, fieldtype, element, qp)
    detJdΩ = element.detJdΩ[qp]
    normal = element.normal[qp]
    f(qp, Nv, normal) * detJdΩ
end
