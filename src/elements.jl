abstract type Element{T, dim} end

# core constructors
Element{T}(shape::Shape{dim}, shape_qr::Shape{dim} = shape) where {T, dim} = SingleBodyElement{T}(shape, shape_qr)
Element{T, dim}(shape::Shape{dim}, shape_qr::Shape{dim} = shape) where {T, dim} = SingleBodyElement{T}(shape, shape_qr)
Element{T, dim}(shape::Shape{shape_dim}, shape_qr::Shape{shape_dim} = shape) where {T, dim, shape_dim} = SingleFaceElement{T, dim}(shape, shape_qr)

Element(::Type{T}, shape::Shape, shape_qr::Shape = shape) where {T} = Element{T}(shape, shape_qr)
Element(shape::Shape, shape_qr::Shape = shape) = Element(Float64, shape, shape_qr)

FaceElement(::Type{T}, shape::Shape{shape_dim}, shape_qr::Shape{shape_dim} = shape) where {T, shape_dim} = SingleFaceElement{T, shape_dim+1}(shape, shape_qr)
FaceElement(shape::Shape, shape_qr::Shape = shape) = FaceElement(Float64, shape, shape_qr)

#################
# SingleElement #
#################

abstract type SingleElement{T, dim} <: Element{T, dim} end

get_shape(elt::SingleElement) = elt.shape
get_shape_qr(elt::SingleElement) = elt.shape_qr
get_dimension(elt::SingleElement{<: Any, dim}) where {dim} = dim
num_nodes(elt::SingleElement) = num_nodes(get_shape(elt))
num_quadpoints(elt::SingleElement) = num_quadpoints(get_shape_qr(elt))
num_dofs(::ScalarField, elt::SingleElement) = num_nodes(elt)
num_dofs(::VectorField, elt::SingleElement) = get_dimension(elt) * num_nodes(elt)

# functions for `Shape`
get_local_coordinates(elt::SingleElement{T}) where {T} = get_local_coordinates(T, get_shape(elt))
quadpoints(elt::SingleElement{T}) where {T} = quadpoints(T, get_shape_qr(elt))
quadweights(elt::SingleElement{T}) where {T} = quadweights(T, get_shape_qr(elt))

dofindices(field::AbstractField, element::SingleElement, I) = dofindices(field, Val(get_dimension(element)), I)

#####################
# SingleBodyElement #
#####################

struct SingleBodyElement{T, dim, S <: Shape{dim}, Sqr <: Shape{dim}, L} <: SingleElement{T, dim}
    shape::S
    shape_qr::Sqr
    N::Vector{SVector{L, T}}
    dNdx::Vector{SVector{L, Vec{dim, T}}}
    detJdΩ::Vector{T}
end

function SingleBodyElement{T}(shape::Shape{dim}, shape_qr::Shape{dim} = shape) where {T, dim}
    L = num_nodes(shape)
    n = num_quadpoints(shape_qr)
    N = zeros(SVector{L, T}, n)
    dNdx = zeros(SVector{L, Vec{dim, T}}, n)
    detJdΩ = zeros(T, n)
    element = SingleBodyElement(shape, shape_qr, N, dNdx, detJdΩ)
    update!(element, get_local_coordinates(shape))
    element
end

function update!(element::SingleBodyElement, xᵢ::AbstractVector{<: Vec})
    @assert num_nodes(element) == length(xᵢ)
    @inbounds for (i, (ξ, w)) in enumerate(zip(quadpoints(element), quadweights(element)))
        Nᵢ, dNᵢdξ = values_gradients(get_shape(element), ξ)
        J = mapreduce(⊗, +, xᵢ, dNᵢdξ)
        element.N[i] = Nᵢ
        element.dNdx[i] = dNᵢdξ .⋅ inv(J)
        element.detJdΩ[i] = w * det(J)
    end
    element
end

#####################
# SingleFaceElement #
#####################

struct SingleFaceElement{T, dim, shape_dim, S <: Shape{shape_dim}, Sqr <: Shape{shape_dim}, L} <: SingleElement{T, dim}
    shape::S
    shape_qr::Sqr
    N::Vector{SVector{L, T}}
    normal::Vector{Vec{dim, T}}
    detJdΩ::Vector{T}
end

function SingleFaceElement{T, dim}(shape::Shape{shape_dim}, shape_qr::Shape{shape_dim} = shape) where {T, dim, shape_dim}
    @assert dim ≥ shape_dim
    L = num_nodes(shape)
    n = num_quadpoints(shape_qr)
    N = zeros(SVector{L, T}, n)
    normal = zeros(Vec{dim, T}, n)
    detJdΩ = zeros(T, n)
    element = SingleFaceElement(shape, shape_qr, N, normal, detJdΩ)
    update!(element, map(x -> Vec{dim}(i -> i ≤ shape_dim ? x[i] : 0), get_local_coordinates(shape)))
    element
end

get_normal(J::Mat{3,2}) = J[:,1] × J[:,2]
get_normal(J::Mat{2,1}) = Vec(J[2,1], -J[1,1])
function update!(element::SingleFaceElement, xᵢ::AbstractVector{<: Vec})
    @assert num_nodes(element) == length(xᵢ)
    @inbounds for (i, (ξ, w)) in enumerate(zip(quadpoints(element), quadweights(element)))
        Nᵢ, dNᵢdξ = values_gradients(get_shape(element), ξ)
        J = mapreduce(⊗, +, xᵢ, dNᵢdξ)
        n = get_normal(J)
        element.N[i] = Nᵢ
        element.normal[i] = normalize(n)
        element.detJdΩ[i] = w * norm(n)
    end
    element
end

###############
# interpolate #
###############

_otimes(x, y) = x * y
_otimes(x::Tensor, y::Tensor) = x ⊗ y

function interpolate(element::SingleElement, uᵢ::AbstractVector, qp::Int)
    @boundscheck 1 ≤ qp ≤ num_quadpoints(element)
    @inbounds begin
        N, dNdx = element.N[qp], element.dNdx[qp]
        u = mapreduce(_otimes, +, uᵢ, N)
        dudx = mapreduce(_otimes, +, uᵢ, dNdx)
    end
    dual_gradient(u, dudx)
end

function interpolate(element::SingleElement, uᵢ::AbstractVector, ξ::Vec)
    N, dNdx = values_gradients(get_shape(element), ξ)
    u = mapreduce(_otimes, +, uᵢ, N)
    dudx = mapreduce(_otimes, +, uᵢ, dNdx)
    dual_gradient(u, dudx)
end

# interpolate `uᵢ` at all quadrature points
function interpolate(element::SingleElement, uᵢ::AbstractVector)
    @assert num_nodes(element) == length(uᵢ)
    mappedarray(1:num_quadpoints(element)) do qp
        @inbounds interpolate(element, uᵢ, qp)
    end
end

interpolate(::ScalarField, element::SingleElement, uᵢ::AbstractVector{<: Real}, args...) = interpolate(element, uᵢ, args...)
interpolate(::VectorField, element::SingleElement{T, dim}, uᵢ::AbstractVector{<: Real}, args...) where {T, dim} = interpolate(element, reinterpret(Vec{dim, T}, uᵢ), args...)

#############
# integrate #
#############

function create_array(::Type{T}, field1::AbstractField, field2::AbstractField, element::SingleElement) where {T}
    m = num_dofs(field1, element)
    n = num_dofs(field2, element)
    zeros(T, m, n)
end
function create_array(::Type{T}, field::AbstractField, element::SingleElement) where {T}
    n = num_dofs(field, element)
    zeros(T, n)
end

# infer_integeltype
infer_integeltype(f, args...) = _infer_integeltype(f, map(typeof, args)...)
@pure _mul_type(::Type{T}, ::Type{U}) where {T, U} = Base._return_type(*, Tuple{T, U})
@pure function _infer_integeltype(f, ::Type{Fld1}, ::Type{Fld2}, ::Type{Elt}) where {Fld1, Fld2, T, Elt <: SingleBodyElement{T}}
    Tv = eltype(Base._return_type(shape_values, Tuple{Fld1, Elt, Int}))
    Tu = eltype(Base._return_type(shape_values, Tuple{Fld2, Elt, Int}))
    Args = Tuple{Int, Tv, Tu}
    ElType = _mul_type(Base._return_type(f, Args), T)
    if ElType == Union{} || ElType == Any
        f(zero_recursive(Args)...) # try run for error case
        error("type inference failed in `infer_integeltype`, consider using inplace version `integrate!`")
    end
    ElType
end
@pure function _infer_integeltype(f, ::Type{Fld}, ::Type{Elt}) where {Fld, T, Elt <: SingleBodyElement{T}}
    Tv = eltype(Base._return_type(shape_values, Tuple{Fld, Elt, Int}))
    Args = Tuple{Int, Tv}
    ElType = _mul_type(Base._return_type(f, Args), T)
    if ElType == Union{} || ElType == Any
        f(zero_recursive(Args)...) # try run for error case
        error("type inference failed in `infer_integeltype`, consider using inplace version `integrate!`")
    end
    ElType
end
@pure function _infer_integeltype(f, ::Type{Fld}, ::Type{Elt}) where {Fld, T, dim, Elt <: SingleFaceElement{T, dim}}
    Tv = eltype(Base._return_type(shape_values, Tuple{Fld, Elt, Int}))
    Args = Tuple{Int, Vec{dim, T}, Tv}
    ElType = _mul_type(Base._return_type(f, Args), T)
    if ElType == Union{} || ElType == Any
        f(zero_recursive(Args)...) # try run for error case
        error("type inference failed in `infer_integeltype`, consider using inplace version `integrate!`")
    end
    ElType
end
@pure function _infer_integeltype(f, ::Type{Elt}) where {T, dim, Elt <: SingleElement{T, dim}}
    Args = Tuple{Int}
    ElType = _mul_type(Base._return_type(f, Args), T)
    if ElType == Union{} || ElType == Any
        f(zero_recursive(Args)...) # try run for error case
        error("type inference failed in `infer_integeltype`, consider using inplace version `integrate!`")
    end
    ElType
end

## integrate without shape values
function integrate(f, element::SingleElement)
    T = infer_integeltype(f, element)
    a = zero(T)
    @inbounds @simd for qp in 1:num_quadpoints(element)
        a += f(qp) * element.detJdΩ[qp]
    end
    a
end

## integrate!
function integrate!(f, A::AbstractMatrix, field1::AbstractField, field2::AbstractField, element::SingleElement)
    for qp in 1:num_quadpoints(element)
        integrate!(f, A, field1, field2, element, qp)
    end
    A
end
function integrate!(f, A::AbstractVector, field::AbstractField, element::SingleElement)
    for qp in 1:num_quadpoints(element)
        integrate!(f, A, field, element, qp)
    end
    A
end

## integrate! at each quadrature point
# SingleBodyElement
function integrate!(f, A::AbstractMatrix, field1::AbstractField, field2::AbstractField, element::SingleBodyElement, qp::Int)
    @boundscheck 1 ≤ qp ≤ num_quadpoints(element)
    @inbounds begin
        v = shape_values(field1, element, qp)
        u = shape_values(field2, element, qp)
        @assert size(A) == (length(v), length(u))
        for j in 1:length(u)
            @simd for i in 1:length(v)
                A[i,j] += f(qp, v[i], u[j]) * element.detJdΩ[qp]
            end
        end
    end
    A
end
function integrate!(f, A::AbstractVector, field::AbstractField, element::SingleBodyElement, qp::Int)
    @boundscheck 1 ≤ qp ≤ num_quadpoints(element)
    @inbounds begin
        v = shape_values(field, element, qp)
        @assert length(A) == length(v)
        @simd for i in 1:length(v)
            A[i] += f(qp, v[i]) * element.detJdΩ[qp]
        end
    end
    A
end

# SingleFaceElement
function integrate!(f, A::AbstractVector, field::AbstractField, element::SingleFaceElement, qp::Int)
    @boundscheck 1 ≤ qp ≤ num_quadpoints(element)
    @inbounds begin
        v = shape_values(field, element, qp)
        @assert length(A) == length(v)
        @simd for i in 1:length(v)
            A[i] += f(qp, element.normal[qp], v[i]) * element.detJdΩ[qp]
        end
    end
    A
end

##############################
# shape values and gradients #
##############################

# ScalarField
shape_values(::ScalarField, ::Val, N::SVector) = N
shape_gradients(::ScalarField, ::Val, dNdx::SVector) = dNdx
# VectorField
@generated function shape_values(::VectorField, ::Val{dim}, N::SVector{L, T}) where {dim, L, T}
    exps = Expr[]
    for k in 1:L, d in 1:dim
        vals = [ifelse(i==d, :(N[$k]), :(zero($T))) for i in 1:dim]
        vec = :(Vec{dim, T}($(vals...)))
        push!(exps, vec)
    end
    quote
        @_inline_meta
        SVector($(exps...))
    end
end
@generated function shape_gradients(::VectorField, ::Val{dim}, dNdx::SVector{L, Vec{dim, T}}) where {dim, L, T}
    exps = Expr[]
    for k in 1:L, d in 1:dim
        grads = [ifelse(i==d, :(dNdx[$k][$j]), :(zero($T))) for i in 1:dim, j in 1:dim]
        mat = :(Mat{dim, dim, T}($(grads...)))
        push!(exps, mat)
    end
    quote
        @_inline_meta
        SVector($(exps...))
    end
end

function shape_values(field::AbstractField, element::SingleBodyElement, qp::Int)
    @_propagate_inbounds_meta
    dim = get_dimension(element)
    map(dual_gradient, shape_values(field, Val(dim), element.N[qp]), shape_gradients(field, Val(dim), element.dNdx[qp]))
end
function shape_values(field::AbstractField, element::SingleFaceElement, qp::Int)
    @_propagate_inbounds_meta
    dim = get_dimension(element)
    shape_values(field, Val(dim), element.N[qp])
end
