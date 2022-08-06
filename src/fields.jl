abstract type Field end

abstract type SingleField{order} <: Field end

struct ScalarField{order} <: SingleField{order} end
struct VectorField{order} <: SingleField{order} end
const Sf = ScalarField
const Vf = VectorField
@pure ScalarField() = ScalarField{nothing}()
@pure VectorField() = VectorField{nothing}()
@pure ScalarField(order::Int) = ScalarField{order}()
@pure VectorField(order::Int) = VectorField{order}()
@pure get_order(::SingleField{order}) where {order} = order::Int

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
dofindices(::VectorField, ::Val{dim}, nodeindex::Int) where {dim} = Index(ntuple(d -> dim*(nodeindex-1) + d, Val(dim)))
function dofindices(fieldtype::VectorField, ::Val{dim}, conn::Index{L}) where {dim, L}
    Index{dim*L}(Iterators.flatten(map(i -> dofindices(fieldtype, Val(dim), i), conn))...)
end