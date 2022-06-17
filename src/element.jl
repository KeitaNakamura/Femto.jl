abstract type Element{T, dim} end

###############
# BodyElement #
###############

struct BodyElement{T, dim, S <: Shape{dim}, num_nodes} <: Element{T, dim}
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

function integrate(f, fieldtype::FieldType, element::BodyElement, eltindex = nothing)
    build = build_body_element_matrix_or_vector(f, eltindex)
    sum(1:num_quadpoints(element)) do qp
        @_inline_meta
        @inbounds begin
            dN = convert_shape_values(fieldtype, element, element.N[qp])
            dNdx = convert_shape_gradients(fieldtype, element, element.dNdx[qp])
            detJdΩ = element.detJdΩ[qp]
        end
        build(f, qp, eltindex, map(dual, dN, dNdx)) * detJdΩ
    end
end

# integrate helpers
@pure function build_body_element_matrix_or_vector(f, eltindex::Nothing)
    nargs = last(methods(f)).nargs - 1
    nargs == 1 && return build_body_element_vector
    nargs == 2 && return build_body_element_matrix
    error("wrong number of arguments in `integrate`, use `(u, v)` for matrix or `(v)` for vector")
end
@pure function build_body_element_matrix_or_vector(f, eltindex)
    nargs = last(methods(f)).nargs - 1
    nargs == 2 && return build_body_element_vector
    nargs == 3 && return build_body_element_matrix
    error("wrong number of arguments in `integrate`, use `(index, u, v)` for matrix or `(index, v)` for vector")
end
@generated function build_body_element_vector(f, qp, eltindex, N::SVector{L}) where {L}
    if eltindex === Nothing
        exps = [:(f(N[$i])) for i in 1:L]
    else
        exps = [:(f(CartesianIndex(qp, eltindex), N[$i])) for i in 1:L]
    end
    quote
        @_inline_meta
        @inbounds SVector{$L}($(exps...))
    end
end
@generated function build_body_element_matrix(f, qp, eltindex, N::SVector{L}) where {L}
    if eltindex === Nothing
        exps = [:(f(N[$j], N[$i])) for i in 1:L, j in 1:L]
    else
        exps = [:(f(CartesianIndex(qp, eltindex), N[$j], N[$i])) for i in 1:L, j in 1:L]
    end
    quote
        @_inline_meta
        @inbounds SMatrix{$L, $L}($(exps...))
    end
end

###############
# FaceElement #
###############

struct FaceElement{T, dim, shape_dim, S <: Shape{shape_dim}, num_nodes} <: Element{T, dim}
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

function integrate(f, fieldtype::FieldType, element::FaceElement, eltindex = nothing)
    build = build_face_element_vector(f, eltindex)
    sum(1:num_quadpoints(element)) do qp
        @_inline_meta
        @inbounds begin
            N = convert_shape_values(fieldtype, element, element.N[qp])
            detJdΩ = element.detJdΩ[qp]
            normal = element.normal[qp]
        end
        build(f, qp, eltindex, N, normal) * detJdΩ
    end
end

# integrate helpers
@pure function build_face_element_vector(f, eltindex::Nothing)
    nargs = last(methods(f)).nargs - 1
    nargs == 2 && return build_face_element_vector
    error("wrong number of arguments in `integrate`, use `(v, normal)`")
end
@pure function build_face_element_vector(f, eltindex)
    nargs = last(methods(f)).nargs - 1
    nargs == 3 && return build_face_element_vector
    error("wrong number of arguments in `integrate`, use `(index, v, normal)`")
end
@generated function build_face_element_vector(f, qp, eltindex, N::SVector{L}, normal) where {L}
    if eltindex === Nothing
        exps = [:(f(N[$i], normal)) for i in 1:L]
    else
        exps = [:(f(CartesianIndex(qp, eltindex), N[$i], normal)) for i in 1:L]
    end
    quote
        @_inline_meta
        @inbounds SVector{$L}($(exps...))
    end
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

##############
# dofindices #
##############

dofindices(::ScalarField, ::Element, conn::Index) = conn

@generated function dofindices(::VectorField, ::Element{<: Any, dim}, conn::Index{L}) where {dim, L}
    exps = [:(starts[$i]+$j) for j in 0:dim-1, i in 1:L]
    quote
        @_inline_meta
        starts = @. dim*(conn - 1) + 1
        @inbounds Index($(exps...))
    end
end

################################################
# convert_shape_values/convert_shape_gradients #
################################################

# ScalarField
convert_shape_values(::ScalarField, ::Element{T}, N::SVector{<: Any, T}) where {T} = N
convert_shape_gradients(::ScalarField, ::Element{T, dim}, dNdx::SVector{<: Any, Vec{dim, T}}) where {T, dim} = dNdx
# VectorField
@generated function convert_shape_values(::VectorField, ::Element{T, dim}, N::SVector{L, T}) where {L, T, dim}
    exps = Expr[]
    for k in 1:L, d in 1:dim
        vals  = [ifelse(i==d, :(N[$k]), :(zero($T))) for i in 1:dim]
        vec = :(Vec{dim, T}($(vals...)))
        push!(exps, vec)
    end
    quote
        @_inline_meta
        @inbounds SVector($(exps...))
    end
end
@generated function convert_shape_gradients(::VectorField, ::Element{T, dim}, dNdx::SVector{L, Vec{dim, T}}) where {L, T, dim}
    exps = Expr[]
    for k in 1:L, d in 1:dim
        grads = [ifelse(i==d, :(dNdx[$k][$j]), :(zero($T))) for i in 1:dim, j in 1:dim]
        mat = :(Mat{dim, dim, T}($(grads...)))
        push!(exps, mat)
    end
    quote
        @_inline_meta
        @inbounds SVector($(exps...))
    end
end

###############
# interpolate #
###############

_otimes(x, y) = x * y
_otimes(x::Vec, y::Vec) = x ⊗ y

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
