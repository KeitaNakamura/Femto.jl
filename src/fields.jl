abstract type Field end

abstract type SingleField{order} <: Field end

struct ScalarField{order} <: SingleField{order} end
struct VectorField{order} <: SingleField{order} end
const Sf = ScalarField
const Vf = VectorField
ScalarField() = ScalarField{nothing}()
VectorField() = VectorField{nothing}()
ScalarField(order::Int) = ScalarField{order}()
VectorField(order::Int) = VectorField{order}()
get_order(::SingleField{order}) where {order} = order

struct MixedField{N, Fields <: NTuple{N, SingleField}} <: Field
    fields::Fields
end
MixedField(fields::SingleField...) = MixedField(fields)
mixed(fields::SingleField...) = MixedField(fields)

## dofindices
# ScalarField
dofindices(::ScalarField, ::Val, nodeindex::Int) = nodeindex
dofindices(::ScalarField, ::Val, conn::Index) = conn
# VectorField
@generated function dofindices(::VectorField, ::Val{dim}, nodeindex::Int) where {dim}
    inds = [:(offset + $d) for d in 1:dim]
    quote
        @_inline_meta
        offset = dim * (nodeindex - 1)
        Index{dim}($(inds...))
    end
end
@generated function dofindices(fieldtype::VectorField, ::Val{dim}, conn::Index{L}) where {dim, L}
    inds = [:(dofindices(fieldtype, Val(dim), conn[$i])...) for i in 1:L]
    quote
        @_inline_meta
        Index{dim*L}($(inds...))
    end
end
