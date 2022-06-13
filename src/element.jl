struct Element{T, dim, F, S <: Shape{dim}, L} # L is number of nodes
    fieldtype::F
    shape::S
    N::Vector{SVector{L, T}}
    dNdx::Vector{SVector{L, Vec{dim, T}}}
    detJdΩ::Vector{T}
end

function Element{T}(fieldtype::FieldType, shape::Shape{dim}) where {T, dim}
    n = num_quadpoints(shape)
    L = num_nodes(shape)
    N = zeros(SVector{L, T}, n)
    dNdx = zeros(SVector{L, Vec{dim, T}}, n)
    detJdΩ = zeros(T, n)
    element = Element(fieldtype, shape, N, dNdx, detJdΩ)
    update!(element, get_local_node_coordinates(shape))
    element
end
Element(fieldtype::FieldType, shape::Shape) = Element{Float64}(fieldtype, shape)

get_shape(elt::Element) = elt.shape
get_fieldtype(elt::Element) = elt.fieldtype
get_dimension(elt::Element) = get_dimension(get_shape(elt))
num_nodes(elt::Element) = num_nodes(get_shape(elt))
num_quadpoints(elt::Element) = num_quadpoints(get_shape(elt))
num_dofs(elt::Element{<: Any, <: Any, ScalarField}) = num_nodes(elt)
num_dofs(elt::Element{<: Any, <: Any, VectorField}) = get_dimension(elt) * num_nodes(elt)

# functions for `Shape`
get_local_node_coordinates(elt::Element{T}) where {T} = get_local_node_coordinates(T, get_shape(elt))
quadpoints(elt::Element{T}) where {T} = quadpoints(T, get_shape(elt))
quadweights(elt::Element{T}) where {T} = quadweights(T, get_shape(elt))

function update!(element::Element, xᵢ::AbstractVector{<: Vec})
    @assert num_nodes(element) == length(xᵢ)
    @inbounds for (i, (ξ, w)) in enumerate(zip(quadpoints(element), quadweights(element)))
        Nᵢ, dNᵢdξ = values_gradients(get_shape(element), ξ)
        J = mapreduce(⊗, +, xᵢ, dNᵢdξ)
        element.N[i] = Nᵢ
        element.dNdx[i] = inv(J') .⋅ dNᵢdξ
        element.detJdΩ[i] = w * det(J)
    end
    element
end

##############
# dofindices #
##############

dofindices(::Element{<: Any, <: Any, ScalarField}, conn::Index) = conn

@generated function dofindices(::Element{<: Any, dim, VectorField}, conn::Index{L}) where {dim, L}
    exps = [:(starts[$i]+$j) for j in 0:dim-1, i in 1:L]
    quote
        @_inline_meta
        starts = @. dim*(conn - 1) + 1
        @inbounds Index($(exps...))
    end
end

#############
# integrate #
#############

function integrate(f, element::Element, eltindex = nothing)
    build = build_element_matrix_or_vector(f, eltindex)
    sum(1:num_quadpoints(element)) do qp
        @_inline_meta
        @inbounds begin
            N, dNdx = transform_field(get_fieldtype(element), element.N[qp], element.dNdx[qp])
            detJdΩ = element.detJdΩ[qp]
        end
        build(f, qp, eltindex, N, dNdx, detJdΩ, Val(num_dofs(element)))
    end
end

# transform (N, dNdx) to given field type
transform_field(::ScalarField, N::SVector{L, T}, dNdx::SVector{L, Vec{dim, T}}) where {L, T, dim} = (N, dNdx)
@generated function transform_field(::VectorField, N::SVector{L, T}, dNdx::SVector{L, Vec{dim, T}}) where {L, T, dim}
    N_exps = Expr[]
    dNdx_exps = Expr[]
    for k in 1:L, d in 1:dim
        vals  = [ifelse(i==d, :(N[$k]), :(zero($T))) for i in 1:dim]
        grads = [ifelse(i==d, :(dNdx[$k][$j]), :(zero($T))) for i in 1:dim, j in 1:dim]
        push!(N_exps, :(Vec{dim, T}($(vals...))))
        push!(dNdx_exps, :(Mat{dim, dim, T}($(grads...))))
    end
    quote
        @_inline_meta
        @inbounds SVector($(N_exps...)), SVector($(dNdx_exps...))
    end
end

# build elementvector/elementmatrix
@generated function build_element_vector(f, qp, eltindex, N, dNdx, detJdΩ, ::Val{L}) where {L}
    if eltindex === Nothing
        exps = [:(f(N[$i], dNdx[$i], detJdΩ)) for i in 1:L]
    else
        exps = [:(f(CartesianIndex(qp, eltindex), N[$i], dNdx[$i], detJdΩ)) for i in 1:L]
    end
    quote
        @_inline_meta
        @inbounds SVector{$L}($(exps...))
    end
end
@generated function build_element_matrix(f, qp, eltindex, N, dNdx, detJdΩ, ::Val{L}) where {L}
    if eltindex === Nothing
        exps = [:(f(N[$j], dNdx[$j], N[$i], dNdx[$i], detJdΩ)) for i in 1:L, j in 1:L]
    else
        exps = [:(f(CartesianIndex(qp, eltindex), N[$j], dNdx[$j], N[$i], dNdx[$i], detJdΩ)) for i in 1:L, j in 1:L]
    end
    quote
        @_inline_meta
        @inbounds SMatrix{$L, $L}($(exps...))
    end
end
@pure function build_element_matrix_or_vector(f, eltindex::Nothing)
    nargs = last(methods(f)).nargs - 1
    nargs == 3 && return build_element_vector
    nargs == 5 && return build_element_matrix
    error("wrong number of arguments in `integrate`, use `(u, ∇u, v, ∇v, dΩ)` for matrix or `(v, ∇v, dΩ)` for vector")
end
@pure function build_element_matrix_or_vector(f, eltindex)
    nargs = last(methods(f)).nargs - 1
    nargs == 4 && return build_element_vector
    nargs == 6 && return build_element_matrix
    error("wrong number of arguments in `integrate`, use `(index, u, ∇u, v, ∇v, dΩ)` for matrix or `(index, v, ∇v, dΩ)` for vector")
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
    u, dudx
end

function interpolate(element::Element, uᵢ::AbstractVector, ξ::Vec)
    N, dNdx = values_gradients(get_shape(element), ξ)
    u = mapreduce(_otimes, +, uᵢ, N)
    dudx = mapreduce(_otimes, +, uᵢ, dNdx)
    u, dudx
end
