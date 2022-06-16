# currently shape must be unique in grid
struct Grid{T, dim, S <: Shape{dim}, L}
    nodes::Vector{Vec{dim, T}}
    shape::S
    connectivities::Vector{Index{L}}
end

get_nodes(grid::Grid) = grid.nodes
get_shape(grid::Grid) = grid.shape
get_connectivities(grid::Grid) = grid.connectivities
get_dimension(grid::Grid{<: Any, dim}) where {dim} = dim

num_nodes(grid::Grid) = length(get_nodes(grid))
num_elements(grid::Grid) = length(get_connectivities(grid))
num_dofs(::ScalarField, grid::Grid) = num_nodes(grid)
num_dofs(::VectorField, grid::Grid) = num_nodes(grid) * get_dimension(grid)

#################
# generate_grid #
#################

_shapetype(::Val{1}) = Line2()
_shapetype(::Val{2}) = Quad4()
_shapetype(::Val{3}) = Hex8()
function _connectivity(I::CartesianIndex{1})
    i = I[1]
    (CartesianIndex(i), CartesianIndex(i+1))
end
function _connectivity(I::CartesianIndex{2})
    i, j = I[1], I[2]
    (CartesianIndex(i,j), CartesianIndex(i+1,j), CartesianIndex(i+1,j+1), CartesianIndex(i,j+1))
end
function _connectivity(I::CartesianIndex{3})
    i, j, k = I[1], I[2], I[3]
    (CartesianIndex(i,j,k), CartesianIndex(i+1,j,k), CartesianIndex(i+1,j+1,k), CartesianIndex(i,j+1,k),
     CartesianIndex(i,j,k+1), CartesianIndex(i+1,j,k+1), CartesianIndex(i+1,j+1,k+1), CartesianIndex(i,j+1,k+1))
end
function generate_grid(axes::Vararg{AbstractVector, dim}) where {dim}
    dims = map(length, axes)
    Eltype = promote_type(map(eltype, axes)...)
    nodes = map(Vec{dim, ifelse(Eltype==Int, Float64, Eltype)}, Iterators.product(axes...))
    connectivities = map(oneunit(CartesianIndex{dim}):CartesianIndex(dims .- 1)) do I
        Index(broadcast(getindex, Ref(LinearIndices(dims)), _connectivity(I)))
    end
    Grid(vec(nodes), _shapetype(Val(dim)), vec(connectivities))
end

##############
# integrate! #
##############

function integrate!(f, A::AbstractMatrix, fieldtype::FieldType, grid::Grid{T}) where {T}
    n = num_dofs(fieldtype, grid)
    @assert size(A) == (n,n)
    element = Element{T}(get_shape(grid))
    for (eltindex, conn) in enumerate(get_connectivities(grid))
        update!(element, get_nodes(grid)[conn])
        Ke = integrate(f, fieldtype, element, eltindex)
        ginds = dofindices(fieldtype, element, conn)
        add!(A, ginds, ginds, Ke)
    end
end

function integrate(f, fieldtype::FieldType, grid::Grid)
    n = num_dofs(fieldtype, grid)
    sizehint = num_nodes(get_shape(grid)) * num_elements(grid)
    A = SparseMatrixIJV(n, n; sizehint)
    integrate!(f, A, fieldtype, grid)
    A
end

#################
# Element state #
#################

function generate_elementstate(::Type{ElementState}, grid::Grid) where {ElementState}
    StructArray{ElementState}(undef, num_quadpoints(get_shape(grid)), num_elements(grid))
end

function update_elementstate!(f, elementstates, grid::Grid)
    for eltindex in 1:num_elements(grid)
    end
end
