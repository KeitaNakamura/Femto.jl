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
    FaceElement,
    SingleBodyElement,
    SingleFaceElement,
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
    get_connectivities,
    get_shape,
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

abstract type AbstractField end
struct ScalarField <: AbstractField end
struct VectorField <: AbstractField end
const Sf = ScalarField
const Vf = VectorField

include("dual.jl")
include("sparse.jl")
include("utils.jl")

include("shapes.jl")
include("elements.jl")
include("grid.jl")
include("solve.jl")

function integrate(f, args...)
    T = infer_integeltype(f, args...)
    A = create_array(T, args...)
    integrate!(f, A, args...)
end

include("gmsh.jl")
include("vtk.jl")

include("precompile.jl")
_precompile_()

end # module
