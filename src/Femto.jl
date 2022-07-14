module Femto

using Reexport
using StaticArrays
using StructArrays
using MappedArrays
@reexport using Tensorial

using Base: @pure, @_inline_meta, @_propagate_inbounds_meta
using LinearAlgebra
using SparseArrays

# reexport
export
    sparse,
    spdiagm

export
    # dual
    ∇,
    # sparse
    SparseMatrixCOO,
    # field
    ScalarField,
    VectorField,
    Sf,
    Vf,
    # Element
    Element,
    BodyElement,
    FaceElement,
    ElementVector,
    ElementMatrix,
    update!,
    integrate,
    interpolate,
    # Shape
    Shape,
    Line2,
    Line3,
    Quad4,
    Quad9,
    Hex8,
    Hex27,
    Tri3,
    Tri6,
    Tet4,
    Tet10,
    # Grid
    Grid,
    num_dofs,
    num_allnodes,
    get_allnodes,
    eachnode,
    eachelement,
    integrate!,
    generate_grid,
    generate_gridset,
    generate_elementstate,
    # solve
    solve!,
    # gmsh
    readgmsh,
    # vtk
    openvtk,
    openpvd,
    closevtk,
    closepvd

const Index{L} = SVector{L, Int}

abstract type FieldType end
struct ScalarField <: FieldType end
struct VectorField <: FieldType end
const Sf = ScalarField
const Vf = VectorField

include("dual.jl")
include("sparse.jl")
include("utils.jl")

include("shapes.jl")
include("elements.jl")
include("grid.jl")
include("solve.jl")

include("gmsh.jl")
include("vtk.jl")

end # module
