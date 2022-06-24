# shape must be unique in grid
struct Grid{T, dim, shape_dim, S <: Shape{shape_dim}, L}
    nodes::Vector{Vec{dim, T}}
    shape::S
    connectivities::Vector{Index{L}}
end

get_nodes(grid::Grid) = grid.nodes
get_shape(grid::Grid) = grid.shape
get_connectivities(grid::Grid) = grid.connectivities
get_dimension(grid::Grid{<: Any, dim}) where {dim} = dim
create_element(grid::Grid{T, dim}) where {T, dim} = Element{T, dim}(get_shape(grid))

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

########################
# integrate/integrate! #
########################

check_size(A::AbstractVector, n::Int) = @assert length(A) == n
check_size(A::AbstractMatrix, n::Int) = @assert size(A) == (n, n)

## integrate!
# with IntegrationStyle
function integrate!(f, A::Union{AbstractVector, AbstractMatrix}, style::IntegrationStyle, fieldtype::FieldType, grid::Grid, element::Element = create_element(grid))
    n = num_dofs(fieldtype, grid)
    check_size(A, n)
    for (eltindex, conn) in enumerate(get_connectivities(grid))
        update!(element, get_nodes(grid)[conn])
        @inline function f′(qp, args...)
            @_propagate_inbounds_meta
            f(CartesianIndex(qp, eltindex), args...)
        end
        Ke = integrate(f′, style, fieldtype, element)
        eltdofs = dofindices(fieldtype, element, conn)
        add!(A, eltdofs, Ke)
    end
    A
end
# without IntegrationStyle
function integrate!(f, A::Union{AbstractVector, AbstractMatrix}, fieldtype::FieldType, grid::Grid, element::Element = create_element(grid))
    integrate!(f, A, TensorStyle(f, element), fieldtype, grid, element)
end

## integrate
# with IntegrationStyle
function integrate(f, style::IntegrationStyle{Matrix}, fieldtype::FieldType, grid::Grid, element::Element = create_element(grid))
    n = num_dofs(fieldtype, grid)
    sizehint = num_dofs(fieldtype, element) * num_elements(grid)
    A = SparseMatrixIJV(n, n; sizehint)
    integrate!(f, A, style, fieldtype, grid, element)
end
function integrate(f, style::IntegrationStyle{Vector}, fieldtype::FieldType, grid::Grid{T}, element::Element = create_element(grid)) where {T}
    n = num_dofs(fieldtype, grid)
    F = zeros(T, n)
    integrate!(f, F, style, fieldtype, grid, element)
end
# without IntegrationStyle
function integrate(f, fieldtype::FieldType, grid::Grid, element::Element = create_element(grid))
    integrate(f, TensorStyle(f, element), fieldtype, grid, element)
end

#########################
# generate_elementstate #
#########################

function generate_elementstate(::Type{ElementState}, grid::Grid) where {ElementState}
    elementstate = StructArray{ElementState}(undef, num_quadpoints(get_shape(grid)), num_elements(grid))
    fillzero!(elementstate)
    if :x in propertynames(elementstate)
        elementstate.x .= interpolate(grid, get_nodes(grid))
    end
    elementstate
end

###############
# interpolate #
###############

# returned mappedarray's size is the same as elementstate matrix
function interpolate(grid::Grid{T}, nodalvalues::AbstractVector) where {T}
    @assert num_nodes(grid) == length(nodalvalues)
    element = Element{T}(get_shape(grid))
    dims = (num_quadpoints(get_shape(grid)), num_elements(grid))
    mappedarray(CartesianIndices(dims)) do I
        qp, eltindex = Tuple(I)
        conn = get_connectivities(grid)[eltindex]
        interpolate(element, nodalvalues[conn], qp)
    end
end

interpolate(::ScalarField, grid::Grid, nodalvalues::AbstractVector{<: Real}) = interpolate(grid, nodalvalues)
interpolate(::VectorField, grid::Grid{T, dim}, nodalvalues::AbstractVector{<: Real}) where {T, dim} = interpolate(grid, reinterpret(Vec{dim, T}, nodalvalues))
