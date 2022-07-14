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
infer_integeltype(f, args...) = _infer_integeltype(f, map(typeof, args)...)
@pure _mul_type(::Type{T}, ::Type{U}) where {T, U} = Base._return_type(*, Tuple{T, U})
@pure function _infer_integeltype(f, ::Type{FT1}, ::Type{FT2}, ::Type{Elt}) where {FT1, FT2, T, Elt <: BodyElement{T}}
    Tv = eltype(Base._return_type(shape_duals, Tuple{FT1, Elt, Int}))
    Tu = eltype(Base._return_type(shape_duals, Tuple{FT2, Elt, Int}))
    ElType = _mul_type(Base._return_type(f, Tuple{Int, Tv, Tu}), T)
    if ElType == Union{} || ElType == Any
        error("type inference failed in `infer_integeltype`, consider using inplace version `integrate!`")
    end
    ElType
end
@pure function _infer_integeltype(f, ::Type{FT}, ::Type{Elt}) where {FT, T, Elt <: BodyElement{T}}
    Tv = eltype(Base._return_type(shape_duals, Tuple{FT, Elt, Int}))
    ElType = _mul_type(Base._return_type(f, Tuple{Int, Tv}), T)
    if ElType == Union{} || ElType == Any
        error("type inference failed in `infer_integeltype`, consider using inplace version `integrate!`")
    end
    ElType
end
@pure function _infer_integeltype(f, ::Type{FT}, ::Type{Elt}) where {FT, T, dim, Elt <: FaceElement{T, dim}}
    Tv = eltype(Base._return_type(shape_values, Tuple{FT, Elt, Int}))
    ElType = _mul_type(Base._return_type(f, Tuple{Int, Vec{dim, T}, Tv}), T)
    if ElType == Union{} || ElType == Any
        error("type inference failed in `infer_integeltype`, consider using inplace version `integrate!`")
    end
    ElType
end

## integrate
function integrate(f, ft1::FieldType, ft2::FieldType, element::SingleElement)
    T = infer_integeltype(f, ft1, ft2, element)
    A = create_elementmatrix(T, ft1, ft2, element)
    integrate!(f, A, ft1, ft2, element)
end
function integrate(f, ft::FieldType, element::SingleElement)
    T = infer_integeltype(f, ft, element)
    A = create_elementvector(T, ft, element)
    integrate!(f, A, ft, element)
end

## integrate!
function integrate!(f, A::AbstractMatrix, ft1::FieldType, ft2::FieldType, element::SingleElement)
    for qp in 1:num_quadpoints(element)
        integrate!(f, A, ft1, ft2, element, qp)
    end
    A
end
function integrate!(f, A::AbstractVector, ft::FieldType, element::SingleElement)
    for qp in 1:num_quadpoints(element)
        integrate!(f, A, ft, element, qp)
    end
    A
end

## integrate! at each quadrature point
# BodyElement
function integrate!(f, A::AbstractMatrix, ft::FT, ::FT, element::BodyElement, qp::Int) where {FT <: FieldType}
    @boundscheck 1 ≤ qp ≤ num_quadpoints(element)
    @inbounds begin
        v = u = shape_duals(ft, element, qp)
        @assert size(A) == (length(v), length(u))
        for j in 1:length(u)
            @simd for i in 1:length(v)
                A[i,j] += f(qp, v[i], u[j]) * element.detJdΩ[qp]
            end
        end
    end
    A
end
function integrate!(f, A::AbstractMatrix, ft1::FieldType, ft2::FieldType, element::BodyElement, qp::Int)
    @boundscheck 1 ≤ qp ≤ num_quadpoints(element)
    @inbounds begin
        v = shape_duals(ft1, element, qp)
        u = shape_duals(ft2, element, qp)
        @assert size(A) == (length(v), length(u))
        for j in 1:length(u)
            @simd for i in 1:length(v)
                A[i,j] += f(qp, v[i], u[j]) * element.detJdΩ[qp]
            end
        end
    end
    A
end
function integrate!(f, A::AbstractVector, ft::FieldType, element::BodyElement, qp::Int)
    @boundscheck 1 ≤ qp ≤ num_quadpoints(element)
    @inbounds begin
        v = shape_duals(ft, element, qp)
        @assert length(A) == length(v)
        @simd for i in 1:length(v)
            A[i] += f(qp, v[i]) * element.detJdΩ[qp]
        end
    end
    A
end

# FaceElement
function integrate!(f, A::AbstractVector, ft::FieldType, element::FaceElement, qp::Int)
    @boundscheck 1 ≤ qp ≤ num_quadpoints(element)
    @inbounds begin
        v = shape_values(ft, element, qp)
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
function shape_duals(ft::FieldType, element::SingleElement, qp::Int)
    @_propagate_inbounds_meta
    map(dual, shape_values(ft, element, qp), shape_gradients(ft, element, qp))
end
