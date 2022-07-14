###########
# Element #
###########

abstract type Element{T, dim} end

Element{T, dim}(shape::Shape{dim}, shape_qr::Shape{dim}=shape) where {T, dim} = BodyElement{T}(shape, shape_qr)
Element{T, dim}(shape::Shape{shape_dim}, shape_qr::Shape{shape_dim}=shape) where {T, dim, shape_dim} = FaceElement{T}(shape, shape_qr)

# use `shape_dim` for `dim` by default
Element{T}(shape::Shape{shape_dim}, shape_qr::Shape{shape_dim}=shape) where {T, shape_dim} = Element{T, shape_dim}(shape, shape_qr)
Element(shape::Shape{shape_dim}, shape_qr::Shape{shape_dim}=shape) where {shape_dim} = Element{Float64, shape_dim}(shape, shape_qr)

###############
# BodyElement #
###############

struct BodyElement{T, dim, S <: Shape{dim}, Sqr <: Shape{dim}, L} <: Element{T, dim}
    shape::S
    shape_qr::Sqr
    N::Vector{SVector{L, T}}
    dNdx::Vector{SVector{L, Vec{dim, T}}}
    detJdΩ::Vector{T}
end

function BodyElement{T}(shape::Shape{dim}, shape_qr::Shape{dim} = shape) where {T, dim}
    L = num_nodes(shape)
    n = num_quadpoints(shape_qr)
    N = zeros(SVector{L, T}, n)
    dNdx = zeros(SVector{L, Vec{dim, T}}, n)
    detJdΩ = zeros(T, n)
    element = BodyElement(shape, shape_qr, N, dNdx, detJdΩ)
    update!(element, get_local_coordinates(shape))
    element
end
BodyElement(shape::Shape, shape_qr::Shape = shape) = BodyElement{Float64}(shape, shape_qr)

function update!(element::BodyElement, xᵢ::AbstractVector{<: Vec})
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

###############
# FaceElement #
###############

struct FaceElement{T, dim, shape_dim, S <: Shape{shape_dim}, Sqr <: Shape{shape_dim}, L} <: Element{T, dim}
    shape::S
    shape_qr::Sqr
    N::Vector{SVector{L, T}}
    normal::Vector{Vec{dim, T}}
    detJdΩ::Vector{T}
end

function FaceElement{T}(shape::Shape{shape_dim}, shape_qr::Shape{shape_dim} = shape) where {T, shape_dim}
    dim = shape_dim + 1
    L = num_nodes(shape)
    n = num_quadpoints(shape_qr)
    N = zeros(SVector{L, T}, n)
    normal = zeros(Vec{dim, T}, n)
    detJdΩ = zeros(T, n)
    element = FaceElement(shape, shape_qr, N, normal, detJdΩ)
    update!(element, map(x -> Vec{dim}(i -> i ≤ shape_dim ? x[i] : 0), get_local_coordinates(shape)))
    element
end
FaceElement(shape::Shape, shape_qr::Shape = shape) = FaceElement{Float64}(shape, shape_qr)

get_normal(J::Mat{3,2}) = J[:,1] × J[:,2]
get_normal(J::Mat{2,1}) = Vec(J[2,1], -J[1,1])
function update!(element::FaceElement, xᵢ::AbstractVector{<: Vec})
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

#################
# SingleElement #
#################

const SingleElement{dim, T} = Union{BodyElement{dim, T}, FaceElement{dim, T}}

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

dofindices(ftype::FieldType, element::SingleElement, I) = dofindices(ftype, Val(get_dimension(element)), I)

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
    dual(u, dudx)
end

function interpolate(element::SingleElement, uᵢ::AbstractVector, ξ::Vec)
    N, dNdx = values_gradients(get_shape(element), ξ)
    u = mapreduce(_otimes, +, uᵢ, N)
    dudx = mapreduce(_otimes, +, uᵢ, dNdx)
    dual(u, dudx)
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

# create_elementmatrix/create_elementvector
function create_elementmatrix(::Type{T}, ft1::FieldType, ft2::FieldType, element::SingleElement) where {T}
    m = num_dofs(ft1, element)
    n = num_dofs(ft2, element)
    zeros(T, m, n)
end
function create_elementvector(::Type{T}, ft::FieldType, element::SingleElement) where {T}
    n = num_dofs(ft, element)
    zeros(T, n)
end

# infer_integeltype
@pure function infer_integtype(types::Type...)
    Base._return_type(build_element, Tuple{types..., Int})
end
function infer_integeltype(f, element::Element, args...)
    T = infer_integtype(typeof(f), map(typeof, args)..., typeof(element))
    if T == Union{} || T == Any || eltype(T) == Union{} || eltype(T) == Any
        first(build_element(f, args..., element, 1)) # try run for error case
        error("type inference failed in `infer_integeltype`, consider using inplace version `integrate!`")
    end
    eltype(T)
end

# integrate!
@inline function integrate!(f, A::AbstractMatrix, ft1::FieldType, ft2::FieldType, element::SingleElement)
    @inbounds for qp in 1:num_quadpoints(element)
        B = build_element(f, ft1, ft2, element, qp)
        @simd for I in eachindex(A, B)
            A[I] += B[I]
        end
    end
    A
end
@inline function integrate!(f, A::AbstractVector, ft::FieldType, element::SingleElement)
    @inbounds for qp in 1:num_quadpoints(element)
        B = build_element(f, ft, element, qp)
        @simd for I in eachindex(A, B)
            A[I] += B[I]
        end
    end
    A
end

# integrate
@inline function integrate(f, ft1::FieldType, ft2::FieldType, element::SingleElement)
    T = infer_integeltype(f, element, ft1, ft2)
    A = create_elementmatrix(T, ft1, ft2, element)
    integrate!(f, A, ft1, ft2, element)
end
@inline function integrate(f, ft::FieldType, element::SingleElement)
    T = infer_integeltype(f, element, ft)
    A = create_elementvector(T, ft, element)
    integrate!(f, A, ft, element)
end

##############################
# shape values and gradients #
##############################

# ScalarField
_shape_values(::ScalarField, ::Val, N::SVector) = N
_shape_gradients(::ScalarField, ::Val, dNdx::SVector) = dNdx
# VectorField
@generated function _shape_values(::VectorField, ::Val{dim}, N::SVector{L, T}) where {dim, L, T}
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
@generated function _shape_gradients(::VectorField, ::Val{dim}, dNdx::SVector{L, Vec{dim, T}}) where {dim, L, T}
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

shape_values(ft::FieldType, element::SingleElement, qp::Int) = (@_propagate_inbounds_meta; _shape_values(ft, Val(get_dimension(element)), element.N[qp]))
shape_gradients(ft::FieldType, element::SingleElement, qp::Int) = (@_propagate_inbounds_meta; _shape_gradients(ft, Val(get_dimension(element)), element.dNdx[qp]))

#################
# build_element #
#################

## build element matrix and vector
# BodyElement
@inline function build_element(f, ft::FT, ::FT, element::BodyElement, qp::Int) where {FT <: FieldType}
    @_propagate_inbounds_meta
    v = u = map(dual, shape_values(ft, element, qp), shape_gradients(ft, element, qp))
    detJdV = element.detJdΩ[qp]
    build_bodyelement_matrix(f, qp, v, u, detJdV)
end
@inline function build_element(f, ft1::FieldType, ft2::FieldType, element::BodyElement, qp::Int)
    @_propagate_inbounds_meta
    v = map(dual, shape_values(ft1, element, qp), shape_gradients(ft1, element, qp))
    u = map(dual, shape_values(ft2, element, qp), shape_gradients(ft2, element, qp))
    detJdV = element.detJdΩ[qp]
    build_bodyelement_matrix(f, qp, v, u, detJdV)
end
@inline function build_element(f, ft::FieldType, element::BodyElement, qp::Int)
    @_propagate_inbounds_meta
    v = map(dual, shape_values(ft, element, qp), shape_gradients(ft, element, qp))
    detJdV = element.detJdΩ[qp]
    build_bodyelement_vector(f, qp, v, detJdV)
end
# FaceElement
@inline function build_element(f, ft::FieldType, element::FaceElement, qp::Int)
    @_propagate_inbounds_meta
    v = shape_values(ft, element, qp)
    detJdA = element.detJdΩ[qp]
    normal = element.normal[qp]
    build_faceelement_vector(f, qp, normal, v, detJdA)
end

# helpers
@inline function build_bodyelement_matrix(f, qp, v::SVector, u::SVector, detJdΩ)
    mappedarray(CartesianIndices((length(v), length(u)))) do I
        @_inline_meta
        i, j = Tuple(I)
        f(qp, v[i], u[j]) * detJdΩ
    end
end
@inline function build_bodyelement_vector(f, qp, v::SVector, detJdΩ)
    mappedarray(v) do vi
        @_inline_meta
        f(qp, vi) * detJdΩ
    end
end
@inline function build_faceelement_vector(f, qp, normal, v::SVector, detJdΩ)
    mappedarray(v) do vi
        @_inline_meta
        f(qp, normal, vi) * detJdΩ
    end
end
