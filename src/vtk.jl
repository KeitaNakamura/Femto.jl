using WriteVTK

to_vtk_celltype(::Line2) = VTKCellTypes.VTK_LINE
to_vtk_celltype(::Line3) = VTKCellTypes.VTK_QUADRATIC_EDGE
to_vtk_celltype(::Line4) = VTKCellTypes.VTK_CUBIC_LINE
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
to_vtk_connectivity(::Line4) = Index(1,2,3,4)
to_vtk_connectivity(::Tri3)  = Index(1,2,3)
to_vtk_connectivity(::Tri6)  = Index(1,2,3,4,6,5)
to_vtk_connectivity(::Quad4) = Index(1,2,3,4)
to_vtk_connectivity(::Quad9) = Index(1,2,3,4,5,6,7,8,9)
to_vtk_connectivity(::Tet4)  = Index(1,2,3,4)
to_vtk_connectivity(::Tet10) = Index(1,2,3,4,5,8,6,7,10,9)
to_vtk_connectivity(::Hex8)  = Index(1,2,3,4,5,6,7,8)
to_vtk_connectivity(::Hex27) = Index(1,2,3,4,5,6,7,8,9,12,14,10,17,19,20,18,11,13,15,16,23,24,22,25,21,26,27)

function _vtk_grid(filename::AbstractString, shape::Shape, grid::Grid{T, dim}; kwargs...) where {T, dim}
    cells = MeshCell[]
    celltype = to_vtk_celltype(shape)
    for eltindex in 1:num_elements(grid)
        conn = get_connectivity(grid, eltindex)
        push!(cells, MeshCell(celltype, conn[to_vtk_connectivity(shape)]))
    end
    order = get_order(shape)
    nodes = get_allnodes(grid, order)
    points = reshape(reinterpret(T, nodes), (dim, length(nodes)))
    vtk_grid(filename, points, cells; kwargs...)
end
WriteVTK.vtk_grid(filename::AbstractString, grid::Grid; kwargs...) = _vtk_grid(filename, get_shape(grid), grid; kwargs...)
WriteVTK.vtk_grid(filename::AbstractString, field::Field, grid::Grid; kwargs...) = _vtk_grid(filename, get_shape(field, grid), grid; kwargs...)

openvtk(args...; kwargs...) = vtk_grid(args...; kwargs...)
openpvd(args...; kwargs...) = paraview_collection(args...; kwargs...)
closevtk(file::WriteVTK.DatasetFile) = vtk_save(file)
closepvd(file::WriteVTK.CollectionFile) = vtk_save(file)
