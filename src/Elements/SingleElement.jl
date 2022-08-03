abstract type SingleElement{T, dim} <: Element{T, dim} end

get_shape(elt::SingleElement) = elt.shape
get_shape_qr(elt::SingleElement) = elt.shape_qr
get_dimension(elt::SingleElement{<: Any, dim}) where {dim} = dim
get_detJdΩ(elt::SingleElement, qp::Int) = (@_propagate_inbounds_meta; elt.detJdΩ[qp])
num_nodes(elt::SingleElement) = num_nodes(get_shape(elt))
num_quadpoints(elt::SingleElement) = num_quadpoints(get_shape_qr(elt))
num_dofs(::ScalarField, elt::SingleElement) = num_nodes(elt)
num_dofs(::VectorField, elt::SingleElement) = get_dimension(elt) * num_nodes(elt)

# functions for `Shape`
get_local_coordinates(elt::SingleElement{T}) where {T} = get_local_coordinates(T, get_shape(elt))
quadpoints(elt::SingleElement{T}) where {T} = quadpoints(T, get_shape_qr(elt))
quadweights(elt::SingleElement{T}) where {T} = quadweights(T, get_shape_qr(elt))

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

get_normal(elt::SingleFaceElement, qp::Int) = (@_propagate_inbounds_meta; elt.normal[qp])

###############
# interpolate #
###############

_otimes(x, y) = x * y
_otimes(x::Tensor, y::Tensor) = x ⊗ y

_reinterpret(::ScalarField, ::Val{dim}, uᵢ::AbstractVector{T}) where {dim, T <: Real} = uᵢ
_reinterpret(::ScalarField, ::Val{dim}, uᵢ::SVector{L, T}) where {dim, L, T <: Real} = uᵢ
_reinterpret(::VectorField, ::Val{dim}, uᵢ::AbstractVector{T}) where {dim, T <: Real} = reinterpret(Vec{dim, T}, uᵢ)
function _reinterpret(::VectorField, ::Val{dim}, uᵢ::SVector{L, T}) where {dim, L, T <: Real}
    n = L ÷ dim
    M = SMatrix{dim, n}(uᵢ)
    SVector(ntuple(j -> Tensor(M[:,j]), Val(n)))
end

function interpolate(element::SingleElement, uᵢ::AbstractVector, qp::Int)
    @assert num_nodes(element) == length(uᵢ)
    @boundscheck @assert 1 ≤ qp ≤ num_quadpoints(element)
    @inbounds N, dNdx = element.N[qp], element.dNdx[qp]
    u = mapreduce(_otimes, +, uᵢ, N)
    dudx = mapreduce(_otimes, +, uᵢ, dNdx)
    dual_gradient(u, dudx)
end

function interpolate(element::SingleElement, uᵢ::AbstractVector, ξ::Vec)
    @assert num_nodes(element) == length(uᵢ)
    N, dNdx = values_gradients(get_shape(element), ξ)
    u = mapreduce(_otimes, +, uᵢ, N)
    dudx = mapreduce(_otimes, +, uᵢ, dNdx)
    dual_gradient(u, dudx)
end

interpolate(fld::SingleField, elt::SingleElement, uᵢ::AbstractVector, qp) =
    interpolate(elt, _reinterpret(fld, Val(get_dimension(elt)), uᵢ), qp)

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

function shape_values(field::SingleField, element::SingleBodyElement, qp::Int)
    @_propagate_inbounds_meta
    dim = get_dimension(element)
    map(dual_gradient, shape_values(field, Val(dim), element.N[qp]), shape_gradients(field, Val(dim), element.dNdx[qp]))
end
function shape_values(field::SingleField, element::SingleFaceElement, qp::Int)
    @_propagate_inbounds_meta
    dim = get_dimension(element)
    shape_values(field, Val(dim), element.N[qp])
end
