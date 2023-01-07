using Femto
using Test
using Random

using SparseArrays
using StaticArrays

using Femto: Line, Quad, Hex, Tri, Tet

function allshapes()
    tuple(
        Line2(),
        Line3(),
        Line4(),
        Quad4(),
        Quad9(),
        Hex8(),
        Hex27(),
        Tri3(),
        Tri6(),
        Tet4(),
        Tet10(),
    )
end

include("dual.jl")
include("utils.jl")

include("shapes.jl")
include("elements.jl")
include("grid.jl")
include("solve.jl")

include("gmsh.jl")

include("examples.jl")
