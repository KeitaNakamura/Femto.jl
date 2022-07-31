module Femto

using Reexport
using StaticArrays
using StructArrays
using MappedArrays
using BlockArrays
@reexport using Tensorial

using Base: @pure, @_inline_meta, @_propagate_inbounds_meta
using LinearAlgebra
using SparseArrays

# reexport
export
    sparse,
    spdiagm

export
    # common
    get_shape,
    num_dofs,
    integrate,
    integrate!,
    interpolate,
    mixed,
    decrease_order,
    # dual
    ∇,
    # sparse
    SparseMatrixCOO,
    # field
    Field,
    ScalarField,
    VectorField,
    Sf,
    Vf,
    # Element
    Element,
    FaceElement,
    SingleElement,
    SingleBodyElement,
    SingleFaceElement,
    MixedElement,
    MixedBodyElement,
    MixedFaceElement,
    update!,
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
    num_allnodes,
    get_allnodes,
    get_connectivities,
    get_dofs,
    get_nodedofs,
    get_elementdofs,
    generate_grid,
    generate_gridset,
    generate_elementstate,
    gridvalues,
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

include("dual.jl")
include("sparse.jl")
include("utils.jl")

include("shapes.jl")
include("fields.jl")
include("Elements/Element.jl")
include("Elements/SingleElement.jl")
include("Elements/MixedElement.jl")
include("Elements/common.jl")
include("grid.jl")
include("integration.jl")
include("solve.jl")

include("gmsh.jl")
include("vtk.jl")

include("precompile.jl")
_precompile_()

end # module
