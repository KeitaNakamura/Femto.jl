import GmshReader

function gmsh_translate(nodeset::GmshReader.NodeSet) where {T}
    dim = nodeset.dim
    map(v -> Vec{dim}(i -> v[i]), nodeset.coord)
end

function gmsh_translate_shape(shape::String)
    shape == "Line 2" && return Line2()
    shape == "Line 3" && return Line3()
    shape == "Triangle 3" && return Tri3()
    shape == "Triangle 6" && return Tri6()
    shape == "Quadrilateral 4" && return Quad4()
    shape == "Quadrilateral 9" && return Quad9()
    shape == "Hexahedron 8" && return Hex8()
    shape == "Hexahedron 27" && return Hex27()
    error("\"$shape\" is not supported yet")
end

function gmsh_translate(elementset::GmshReader.ElementSet)
    shape = gmsh_translate_shape(elementset.elementname)
    connectivities = map(Index{num_nodes(shape)}, elementset.connectivities)
    shape, connectivities
end

function gmsh_translate(phygroup::GmshReader.PhysicalGroup)
    # nodes
    nodeindices = phygroup.nodeset.nodetags
    # elements
    dict = Dict{Shape, Vector{<: Index}}()
    for entitiy in phygroup.entities
        for elementset in entitiy
            shape, connectivities = gmsh_translate(elementset)
            append!(get!(dict, shape, Index{num_nodes(shape)}[]), connectivities)
        end
    end
    (nodeindices, only(dict)...) # support only unique shape
end

function gmsh_translate(gmshfile::GmshReader.GmshFile)
    dim = gmshfile.nodeset.dim
    # nodes
    nodes = gmsh_translate(gmshfile.nodeset)
    # elements
    Dict{String, Grid{Float64, dim}}(map(collect(gmshfile.physicalgroups)) do (name, phygroup)
        nodeindices, shape, connectivities = gmsh_translate(phygroup)
        name => Grid(nodes, nodeindices, shape, connectivities)
    end)
end

function readgmsh(filename::String)
    file = GmshReader.readgmsh(filename; fixsurface = true)
    gmsh_translate(file)
end
