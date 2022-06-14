module Femto

using Reexport
using StaticArrays
using StructArrays
@reexport using Tensorial

using Base: @pure, @_inline_meta
using SparseArrays

export
    # dual
    ∇,
    # field
    ScalarField,
    VectorField,
    # Element
    Element,
    update!,
    integrate,
    # Shape
    Quad4,
    Tri6,
    # Grid
    Grid,
    integrate!

const Index{L} = SVector{L, Int}

abstract type FieldType end
struct ScalarField <: FieldType end
struct VectorField <: FieldType end

include("dual.jl")

include("shapes.jl")
include("element.jl")
include("sparse.jl")
include("grid.jl")

end # module
