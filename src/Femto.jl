module Femto

using Reexport
using StaticArrays
using StructArrays
using MappedArrays
@reexport using Tensorial

using Base: @pure, @_inline_meta, @_propagate_inbounds_meta
using LinearAlgebra
using SparseArrays

export
    # dual
    ∇,
    # field
    ScalarField,
    VectorField,
    # Element
    Element,
    BodyElement,
    FaceElement,
    update!,
    integrate,
    interpolate,
    # Shape
    Line2,
    Quad4,
    Hex8,
    Tri6,
    # Grid
    Grid,
    integrate!,
    generate_grid,
    generate_elementstate

const Index{L} = SVector{L, Int}

abstract type FieldType end
struct ScalarField <: FieldType end
struct VectorField <: FieldType end

include("dual.jl")

include("shapes.jl")
include("elements.jl")
include("sparse.jl")
include("grid.jl")
include("utils.jl")

include("integration.jl")


end # module
