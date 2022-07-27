import GmshReader

function from_gmsh(nodeset::GmshReader.NodeSet) where {T}
    dim = nodeset.dim
    map(v -> Vec{dim}(i -> v[i]), nodeset.coord)
end

function from_gmsh_shape(shape::String)
    shape == "Line 2" && return Line2()
    shape == "Line 3" && return Line3()
    shape == "Triangle 3" && return Tri3()
    shape == "Triangle 6" && return Tri6()
    shape == "Tetrahedron 4" && return Tet4()
    shape == "Tetrahedron 10" && return Tet10()
    shape == "Quadrilateral 4" && return Quad4()
    shape == "Quadrilateral 9" && return Quad9()
    shape == "Hexahedron 8" && return Hex8()
    shape == "Hexahedron 27" && return Hex27()
    error("\"$shape\" is not supported yet")
end

from_gmsh_connectivity(::Line2) = Index(1,2)
from_gmsh_connectivity(::Line3) = Index(1,2,3)
from_gmsh_connectivity(::Tri3)  = Index(1,2,3)
from_gmsh_connectivity(::Tri6)  = Index(1,2,3,4,6,5)
from_gmsh_connectivity(::Tet4)  = Index(1,2,3,4)
from_gmsh_connectivity(::Tet10) = Index(1,2,3,4,5,7,8,6,9,10)
from_gmsh_connectivity(::Quad4) = Index(1,2,3,4)
from_gmsh_connectivity(::Quad9) = Index(1,2,3,4,5,6,7,8,9)
from_gmsh_connectivity(::Hex8)  = Index(1,2,3,4,5,6,7,8)
from_gmsh_connectivity(::Hex27) = Index(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27)

function from_gmsh(elementset::GmshReader.ElementSet)
    shape = from_gmsh_shape(elementset.elementname)
    connectivities = map(elementset.connectivities) do conn
        @assert num_nodes(shape) == length(conn)
        conn[from_gmsh_connectivity(shape)]
    end
    shape, connectivities
end

function from_gmsh(phygroup::GmshReader.PhysicalGroup)
    # nodes
    nodeindices = phygroup.nodeset.nodetags
    # elements
    dict = Dict{Shape, Vector{<: Index}}()
    for entitiy in phygroup.entities
        for elementset in entitiy
            shape, connectivities = from_gmsh(elementset)
            append!(get!(dict, shape, Index{num_nodes(shape)}[]), connectivities)
        end
    end
    (nodeindices, only(dict)...) # support only unique shape
end

function from_gmsh(gmshfile::GmshReader.GmshFile)
    dim = gmshfile.nodeset.dim
    # nodes
    nodes = from_gmsh(gmshfile.nodeset)
    # physicalgroups
    physicalgroups = Dict(Iterators.map(gmshfile.physicalgroups) do (name, phygroup)
        nodeindices, shape, connectivities = from_gmsh(phygroup)
        name => (; nodeindices, shape, connectivities)
    end)
    bodies = Dict(Iterators.filter(p -> get_dimension(p.second.shape) == dim, physicalgroups))
    faces = Dict(Iterators.filter(p -> get_dimension(p.second.shape) != dim, physicalgroups))
    nodes, bodies, faces
end

function create_gridset(nodes::Vector{<: Vec{dim}}, phygroup::Dict) where {dim}
    Dict{String, Grid{Float64, dim}}(Iterators.map(phygroup) do (name, group)
        name => Grid(nodes, group.shape, group.connectivities, group.nodeindices)
    end)
end

function readgmsh(filename::String)
    file = GmshReader.readgmsh(filename; fixsurface = true)
    nodes, bodies, faces = from_gmsh(file)
    merge(create_gridset(nodes, bodies), create_gridset(nodes, faces))
end
