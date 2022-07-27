using WriteVTK

to_vtk_celltype(::Line2) = VTKCellTypes.VTK_LINE
to_vtk_celltype(::Line3) = VTKCellTypes.VTK_QUADRATIC_EDGE
to_vtk_celltype(::Tri3)  = VTKCellTypes.VTK_TRIANGLE
to_vtk_celltype(::Tri6)  = VTKCellTypes.VTK_QUADRATIC_TRIANGLE
to_vtk_celltype(::Quad4) = VTKCellTypes.VTK_QUAD
to_vtk_celltype(::Quad9) = VTKCellTypes.VTK_BIQUADRATIC_QUAD
to_vtk_celltype(::Tet4)  = VTKCellTypes.VTK_TETRA
to_vtk_celltype(::Tet10) = VTKCellTypes.VTK_QUADRATIC_TETRA
to_vtk_celltype(::Hex8)  = VTKCellTypes.VTK_HEXAHEDRON
to_vtk_celltype(::Hex27) = VTKCellTypes.VTK_TRIQUADRATIC_HEXAHEDRON

to_vtk_connectivity(::Line2) = Index(1,2)
to_vtk_connectivity(::Line3) = Index(1,2,3)
to_vtk_connectivity(::Tri3)  = Index(1,2,3)
to_vtk_connectivity(::Tri6)  = Index(1,2,3,4,6,5)
to_vtk_connectivity(::Quad4) = Index(1,2,3,4)
to_vtk_connectivity(::Quad9) = Index(1,2,3,4,5,6,7,8,9)
to_vtk_connectivity(::Tet4)  = Index(1,2,3,4)
to_vtk_connectivity(::Tet10) = Index(1,2,3,4,5,8,6,7,10,9)
to_vtk_connectivity(::Hex8)  = Index(1,2,3,4,5,6,7,8)
to_vtk_connectivity(::Hex27) = Index(1,2,3,4,5,6,7,8,9,12,14,10,17,19,20,18,11,13,15,16,23,24,22,25,21,26,27)

function WriteVTK.vtk_grid(filename::AbstractString, grid::Grid{T, dim}; kwargs...) where {T, dim}
    cells = MeshCell[]
    shape = get_shape(grid)
    celltype = to_vtk_celltype(shape)
    for conn in get_connectivities(grid)
        push!(cells, MeshCell(celltype, conn[to_vtk_connectivity(shape)]))
    end
    points = reshape(reinterpret(T, get_allnodes(grid)), (dim, num_allnodes(grid)))
    vtk_grid(filename, points, cells; kwargs...)
end

openvtk(args...; kwargs...) = vtk_grid(args...; kwargs...)
openpvd(args...; kwargs...) = paraview_collection(args...; kwargs...)
closevtk(file::WriteVTK.DatasetFile) = vtk_save(file)
closepvd(file::WriteVTK.CollectionFile) = vtk_save(file)
