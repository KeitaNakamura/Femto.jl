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
    Line3,
    Quad4,
    Quad9,
    Hex8,
    Hex27,
    Tri3,
    Tri6,
    # Grid
    Grid,
    create_globalvector,
    create_globalmatrix,
    integrate!,
    generate_grid,
    generate_elementstate,
    # solve
    solve!,
    create_solutionvector

const Index{L} = SVector{L, Int}

abstract type FieldType end
struct ScalarField <: FieldType end
struct VectorField <: FieldType end

include("dual.jl")
include("sparse.jl")
include("utils.jl")

include("shapes.jl")
include("elements.jl")
include("integration.jl")

include("grid.jl")
include("solve.jl")

end # module
