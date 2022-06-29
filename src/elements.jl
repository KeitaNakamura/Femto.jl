abstract type Element{T, dim, shape, num_nodes} end

###############
# BodyElement #
###############

struct BodyElement{T, dim, S <: Shape{dim}, num_nodes} <: Element{T, dim, S, num_nodes}
    shape::S
    N::Vector{SVector{num_nodes, T}}
    dNdx::Vector{SVector{num_nodes, Vec{dim, T}}}
    detJdΩ::Vector{T}
end

function BodyElement{T}(shape::Shape{dim}) where {T, dim}
    n = num_quadpoints(shape)
    L = num_nodes(shape)
    N = zeros(SVector{L, T}, n)
    dNdx = zeros(SVector{L, Vec{dim, T}}, n)
    detJdΩ = zeros(T, n)
    element = BodyElement(shape, N, dNdx, detJdΩ)
    update!(element, get_local_node_coordinates(shape))
    element
end
BodyElement(shape::Shape) = BodyElement{Float64}(shape)

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

struct FaceElement{T, dim, shape_dim, S <: Shape{shape_dim}, num_nodes} <: Element{T, dim, S, num_nodes}
    shape::S
    N::Vector{SVector{num_nodes, T}}
    normal::Vector{Vec{dim, T}}
    detJdΩ::Vector{T}
end

function FaceElement{T}(shape::Shape{shape_dim}) where {T, shape_dim}
    dim = shape_dim + 1
    n = num_quadpoints(shape)
    L = num_nodes(shape)
    N = zeros(SVector{L, T}, n)
    normal = zeros(Vec{dim, T}, n)
    detJdΩ = zeros(T, n)
    FaceElement(shape, N, normal, detJdΩ)
end
FaceElement(shape::Shape) = FaceElement{Float64}(shape)

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

###########
# Element #
###########

Element{T, dim}(shape::Shape{dim}) where {T, dim} = BodyElement{T}(shape)
Element{T, dim}(shape::Shape{shape_dim}) where {T, dim, shape_dim} = FaceElement{T}(shape)

# use `shape_dim` for `dim` by default
Element{T}(shape::Shape{shape_dim}) where {T, shape_dim} = Element{T, shape_dim}(shape)
Element(shape::Shape{shape_dim}) where {shape_dim} = Element{Float64, shape_dim}(shape)

get_shape(elt::Element) = elt.shape
get_dimension(elt::Element{<: Any, dim}) where {dim} = dim
num_nodes(elt::Element) = num_nodes(get_shape(elt))
num_quadpoints(elt::Element) = num_quadpoints(get_shape(elt))
num_dofs(::ScalarField, elt::Element) = num_nodes(elt)
num_dofs(::VectorField, elt::Element) = get_dimension(elt) * num_nodes(elt)

# functions for `Shape`
get_local_node_coordinates(elt::Element{T}) where {T} = get_local_node_coordinates(T, get_shape(elt))
quadpoints(elt::Element{T}) where {T} = quadpoints(T, get_shape(elt))
quadweights(elt::Element{T}) where {T} = quadweights(T, get_shape(elt))

dofindices(ftype::FieldType, element::Element, I) = dofindices(ftype, Val(get_dimension(element)), I)

###############
# interpolate #
###############

_otimes(x, y) = x * y
_otimes(x::Tensor, y::Tensor) = x ⊗ y

function interpolate(element::Element, uᵢ::AbstractVector, qp::Int)
    @boundscheck 1 ≤ qp ≤ num_quadpoints(element)
    @inbounds begin
        N, dNdx = element.N[qp], element.dNdx[qp]
        u = mapreduce(_otimes, +, uᵢ, N)
        dudx = mapreduce(_otimes, +, uᵢ, dNdx)
    end
    dual(u, dudx)
end

function interpolate(element::Element, uᵢ::AbstractVector, ξ::Vec)
    N, dNdx = values_gradients(get_shape(element), ξ)
    u = mapreduce(_otimes, +, uᵢ, N)
    dudx = mapreduce(_otimes, +, uᵢ, dNdx)
    dual(u, dudx)
end

# interpolate `uᵢ` at all quadrature points
function interpolate(element::Element, uᵢ::AbstractVector)
    @assert num_nodes(element) == length(uᵢ)
    mappedarray(1:num_quadpoints(element)) do qp
        @inbounds interpolate(element, uᵢ, qp)
    end
end

interpolate(::ScalarField, element::Element, uᵢ::AbstractVector{<: Real}, args...) = interpolate(element, uᵢ, args...)
interpolate(::VectorField, element::Element{T, dim}, uᵢ::AbstractVector{<: Real}, args...) where {T, dim} = interpolate(element, reinterpret(Vec{dim, T}, uᵢ), args...)

#############
# integrate #
#############

struct ElementArrayType{VectorOrMatrix}
    ElementArrayType{VectorOrMatrix}() where {VectorOrMatrix} = new{VectorOrMatrix::Union{Type{Vector}, Type{Matrix}}}()
end
const ElementMatrix = ElementArrayType{Matrix}
const ElementVector = ElementArrayType{Vector}

function integrate(f, fieldtype::FieldType, element::Element)
    arrtype = ElementArrayType(f, element)
    integrate(f, arrtype, fieldtype, element)
end

function integrate(f, arraytype::ElementArrayType, fieldtype::FieldType, element::Element)
    sum(1:num_quadpoints(element)) do qp
        @_inline_meta
        @inbounds build_element(f, arraytype, fieldtype, element, qp)
    end
end

## constructors
ElementArrayType(f, element::Element) = ElementArrayType(f, typeof(element))
@pure function ElementArrayType(f, ::Type{<: BodyElement})
    nargs = first(methods(f)).nargs - 1
    nargs == 2 && return ElementVector()
    nargs == 3 && return ElementMatrix()
    error("wrong number of arguments in `integrate`, use `(index, u, v)` for matrix or `(index, v)` for vector")
end
@pure function ElementArrayType(f, ::Type{<: FaceElement})
    nargs = first(methods(f)).nargs - 1
    nargs == 3 && return ElementVector()
    error("wrong number of arguments in `integrate`, use `(index, v, normal)`")
end

##############################
# shape values and gradients #
##############################

# ScalarField
shape_values(::ScalarField, element::Element, qp::Int) = (@_propagate_inbounds_meta; element.N[qp])
shape_gradients(::ScalarField, element::Element, qp::Int) = (@_propagate_inbounds_meta; element.dNdx[qp])
# VectorField
@generated function shape_values(::VectorField, element::Element{T, dim, <: Any, L}, qp::Int) where {T, dim, L}
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
@generated function shape_gradients(::VectorField, element::Element{T, dim, <: Any, L}, qp::Int) where {T, dim, L}
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

#################
# build_element #
#################

## build element matrix and vector
# BodyElement
@inline function build_element(f, ::ElementArrayType{VectorOrMatrix}, fieldtype::FieldType, element::Element, qp::Int) where {VectorOrMatrix}
    @_propagate_inbounds_meta
    N = shape_values(fieldtype, element, qp)
    dNdx = shape_gradients(fieldtype, element, qp)
    detJdΩ = element.detJdΩ[qp]
    u = v = map(dual, N, dNdx)
    VectorOrMatrix == Vector && return build_element_vector(f, qp, v)    * detJdΩ
    VectorOrMatrix == Matrix && return build_element_matrix(f, qp, v, u) * detJdΩ
    error("unreachable")
end
# FaceElement
@inline function build_element(f, ::ElementArrayType{Vector}, fieldtype::FieldType, element::FaceElement, qp::Int)
    @_propagate_inbounds_meta
    v = shape_values(fieldtype, element, qp)
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
