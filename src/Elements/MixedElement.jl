abstract type MixedElement{T, dim, N} <: Element{T, dim} end

MixedElement(elements::SingleBodyElement...) = MixedBodyElement(elements)
MixedElement(elements::SingleFaceElement...) = MixedFaceElement(elements)

function get_detJdΩ(mixed::MixedElement, qp::Int)
    element = argmax(num_nodes, mixed.elements)
    get_detJdΩ(element, qp)
end
function get_normal(mixed::MixedElement, qp::Int)
    element = argmax(num_nodes, mixed.elements)
    get_normal(element, qp)
end
get_shape_qr(mixed::MixedElement) = get_shape_qr(mixed.elements[1])

@generated function num_dofs(fld::MixedField{N}, mixed::MixedElement{<: Any, <: Any, N}) where {N}
    exps = [:(num_dofs(fld.fields[$i], mixed.elements[$i])) for i in 1:N]
    quote
        +($(exps...))
    end
end
function num_quadpoints(mixed::MixedElement)
    num_quadpoints(get_shape_qr(mixed))
end

@pure default_quadrature_shape(shapes::Tuple{Vararg{Shape}}) = shapes[argmax(map(get_order, shapes))]

####################
# MixedBodyElement #
####################

struct MixedBodyElement{T, dim, N, Sqr <: Shape{dim}, Elts <: NTuple{N, SingleBodyElement{T, dim, <: Shape{dim}, Sqr}}} <: MixedElement{T, dim, N}
    elements::Elts
end

function MixedBodyElement{T}(shapes::Tuple{Vararg{Shape}}, shape_qr::Shape = default_quadrature_shape(shapes)) where {T}
    MixedBodyElement(map(shape -> Element{T}(shape, shape_qr), shapes))
end

function Base.show(io::IO, mixed::MixedBodyElement)
    print(io, MixedBodyElement, (map(get_shape, mixed.elements), get_shape_qr(mixed)))
end

####################
# MixedFaceElement #
####################

struct MixedFaceElement{T, dim, N, shape_dim, Sqr <: Shape{shape_dim}, Elts <: NTuple{N, SingleFaceElement{T, dim, shape_dim, <: Shape{shape_dim}, Sqr}}} <: MixedElement{T, dim, N}
    elements::Elts
end

function MixedFaceElement{T, dim}(shapes::Tuple{Vararg{Shape}}, shape_qr::Shape = default_quadrature_shape(shapes)) where {T, dim}
    MixedFaceElement(map(shape -> Element{T, dim}(shape, shape_qr), shapes))
end

function Base.show(io::IO, mixed::MixedFaceElement)
    print(io, MixedFaceElement, (map(get_shape, mixed.elements), get_shape_qr(mixed)))
end

################
# shape values #
################

@generated function shape_values(fld::MixedField{N}, mixed::MixedElement{<: Any, <: Any, N}, qp::Int) where {N}
    exps = [:(shape_values(fld.fields[$i], mixed.elements[$i], qp)) for i in 1:N]
    quote
        @_propagate_inbounds_meta
        complement_shape_values($(exps...))
    end
end
@generated function complement_shape_values(Ns::Vararg{SVector, L}) where {L}
    exps = Expr[]
    for i in 1:L
        for j in 1:length(Ns[i])
            tup = Expr(:tuple)
            for k in 1:L
                if k == i
                    push!(tup.args, :(Ns[$k][$j]))
                else
                    push!(tup.args, :(zero($(eltype(Ns[k])))))
                end
            end
            push!(exps, tup)
        end
    end
    quote
        @_inline_meta
        @inbounds SVector($(exps...))
    end
end

###########
# update! #
###########

function update!(mixed::MixedElement, xᵢ::AbstractVector{<: Vec})
    map(elt -> update!(elt, xᵢ[SOneTo(num_nodes(elt))]), mixed.elements)
end

###############
# interpolate #
###############

@pure function _sranges(n::Tuple{Vararg{Int}})
    stop = cumsum(n)
    start = @. stop - n + 1
    @. StaticArrays.SUnitRange(start, stop)
end

@inline function interpolate(field::MixedField, element::MixedElement, Uᵢ::AbstractVector{<: Real}, qp)
    @assert num_dofs(field, element) == length(Uᵢ)
    rngs = _sranges(map(num_dofs, field.fields, element.elements))
    map(field.fields, element.elements, rngs) do fld, elt, rng
        uᵢ = _reinterpret(fld, Val(get_dimension(elt)), Uᵢ[rng])
        interpolate(elt, uᵢ, qp)
    end
end
