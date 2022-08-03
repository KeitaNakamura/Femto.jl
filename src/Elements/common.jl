const BodyElementLike{T, dim} = Union{SingleBodyElement{T, dim}, MixedBodyElement{T, dim}}
const FaceElementLike{T, dim} = Union{SingleFaceElement{T, dim}, MixedFaceElement{T, dim}}

##########
# common #
##########

function create_matrix(::Type{T}, field::Field, element::Element) where {T}
    m = n = num_dofs(field, element)
    zeros(T, m, n)
end
function create_vector(::Type{T}, field::Field, element::Element) where {T}
    n = num_dofs(field, element)
    zeros(T, n)
end

# infer
infer_integrate_matrix_eltype(f, args...) = _infer_integrate_matrix_eltype(f, map(typeof, args)...)
infer_integrate_vector_eltype(f, args...) = _infer_integrate_vector_eltype(f, map(typeof, args)...)
infer_integrate_eltype(f, args...) = _infer_integrate_eltype(f, map(typeof, args)...)
@pure _mul_type(::Type{T}, ::Type{U}) where {T, U} = Base._return_type(*, Tuple{T, U})
@pure function _infer_integrate_matrix_eltype(f, ::Type{Fld}, ::Type{Elt}) where {Fld, T, Elt <: BodyElementLike{T}}
    Tv = Tu = eltype(Base._return_type(shape_values, Tuple{Fld, Elt, Int}))
    Args = Tuple{Int, Tv, Tu}
    ElType = _mul_type(Base._return_type(f, Args), T)
    if ElType == Union{} || ElType == Any
        f(zero_recursive(Args)...) # try run for error case
        error("type inference failed in `integrate`, consider using inplace version `integrate!`")
    end
    ElType
end
@pure function _infer_integrate_vector_eltype(f, ::Type{Fld}, ::Type{Elt}) where {Fld, T, Elt <: BodyElementLike{T}}
    Tv = eltype(Base._return_type(shape_values, Tuple{Fld, Elt, Int}))
    Args = Tuple{Int, Tv}
    ElType = _mul_type(Base._return_type(f, Args), T)
    if ElType == Union{} || ElType == Any
        f(zero_recursive(Args)...) # try run for error case
        error("type inference failed in `integrate`, consider using inplace version `integrate!`")
    end
    ElType
end
@pure function _infer_integrate_vector_eltype(f, ::Type{Fld}, ::Type{Elt}) where {Fld, T, dim, Elt <: FaceElementLike{T, dim}}
    Tv = eltype(Base._return_type(shape_values, Tuple{Fld, Elt, Int}))
    Args = Tuple{Int, Vec{dim, T}, Tv}
    ElType = _mul_type(Base._return_type(f, Args), T)
    if ElType == Union{} || ElType == Any
        f(zero_recursive(Args)...) # try run for error case
        error("type inference failed in `integrate`, consider using inplace version `integrate!`")
    end
    ElType
end
@pure function _infer_integrate_eltype(f, ::Type{Elt}) where {T, dim, Elt <: Element{T, dim}}
    Args = Tuple{Int}
    ElType = _mul_type(Base._return_type(f, Args), T)
    if ElType == Union{} || ElType == Any
        f(zero_recursive(Args)...) # try run for error case
        error("type inference failed in `integrate`, consider using inplace version `integrate!`")
    end
    ElType
end

#############
# integrate #
#############

## integrate without shape values
function integrate(f, element::Element)
    T = infer_integrate_eltype(f, element)
    a = zero(T)
    @inbounds @simd for qp in 1:num_quadpoints(element)
        a += f(qp) * element.detJdΩ[qp]
    end
    a
end

## integrate!
function integrate!(f, A::AbstractArray, field::Field, element::Element)
    for qp in 1:num_quadpoints(element)
        integrate!(f, A, field, element, qp)
    end
    A
end

## integrate! at each quadrature point
# BodyElementLike
function integrate!(f, A::AbstractMatrix, field::Field, element::BodyElementLike, qp::Int)
    @boundscheck 1 ≤ qp ≤ num_quadpoints(element)
    @inbounds begin
        v = u = shape_values(field, element, qp)
        @assert size(A) == (length(v), length(u))
        for j in 1:length(u)
            @simd for i in 1:length(v)
                A[i,j] += f(qp, v[i], u[j]) * get_detJdΩ(element, qp)
            end
        end
    end
    A
end
function integrate!(f, A::AbstractVector, field::Field, element::BodyElementLike, qp::Int)
    @boundscheck 1 ≤ qp ≤ num_quadpoints(element)
    @inbounds begin
        v = shape_values(field, element, qp)
        @assert length(A) == length(v)
        @simd for i in 1:length(v)
            A[i] += f(qp, v[i]) * get_detJdΩ(element, qp)
        end
    end
    A
end

# FaceElementLike
function integrate!(f, A::AbstractVector, field::Field, element::FaceElementLike, qp::Int)
    @boundscheck 1 ≤ qp ≤ num_quadpoints(element)
    @inbounds begin
        v = shape_values(field, element, qp)
        @assert length(A) == length(v)
        @simd for i in 1:length(v)
            A[i] += f(qp, get_normal(element, qp), v[i]) * get_detJdΩ(element, qp)
        end
    end
    A
end

function integrate(f, field::Field, element::Element)
    F = integrate_function(f, typeof(element))
    F(f, field, element)
end

function integrate_function(f, ::Type{Elt}) where {Elt <: Element}
    nargs = first(methods(f)).nargs - 1
    # integrate
    Elt <: BodyElementLike && nargs == 3 && return integrate_matrix
    Elt <: BodyElementLike && nargs == 2 && return integrate_vector
    Elt <: FaceElementLike && nargs == 3 && return integrate_vector
    # errors
    Elt <: BodyElementLike && error("wrong number of arguments in `integrate`, use `(index, v, u)` for matrix or `(index, v)` for vector")
    Elt <: FaceElementLike && error("wrong number of arguments in `integrate`, use `(index, normal, v)`")
    error("unreachable")
end

###############
# interpolate #
###############

function interpolate(field::Field, element::Element, uᵢ::AbstractVector)
    @assert num_dofs(field, element) == length(uᵢ)
    mappedarray(1:num_quadpoints(element)) do qp
        @inbounds interpolate(field, element, uᵢ, qp)
    end
end
