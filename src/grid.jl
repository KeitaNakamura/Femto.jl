# shape must be unique in grid
struct Grid{T, dim, shape_dim, S <: Shape{shape_dim}, L}
    shape::S
    nodes::Vector{Vec{dim, T}}
    connectivities::Vector{Index{L}}
    nodeindices::Vector{Int}
end

function Grid(shape::Shape, nodes::Vector{<: Vec}, connectivities::Vector{<: Index})
    Grid(shape, nodes, connectivities, collect(1:length(nodes)))
end

# useful to extend the dimension (e.g., dim=2 -> dim=3)
function Grid{T, dim}(grid::Grid) where {T, dim}
    shape = get_shape(grid)
    nodes = map(x -> Vec{dim, T}(Tensorial.resizedim(x, Val(dim))), get_allnodes(grid))
    connectivities = get_connectivities(grid)
    nodeindices = get_nodeindices(grid)
    Grid(shape, nodes, connectivities, nodeindices)
end
Grid{T, dim}(grid::Grid{T, dim}) where {T, dim} = grid

#########
# utils #
#########

get_allnodes(grid::Grid) = grid.nodes
get_nodeindices(grid::Grid) = grid.nodeindices
get_shape(grid::Grid) = grid.shape
get_connectivities(grid::Grid) = grid.connectivities
get_dimension(grid::Grid{<: Any, dim}) where {dim} = dim
get_elementtype(grid::Grid) = Base._return_type(create_element, Tuple{typeof(grid)})
create_element(grid::Grid{T, dim}) where {T, dim} = Element{T, dim}(get_shape(grid))

num_allnodes(grid::Grid) = length(get_allnodes(grid))
num_elements(grid::Grid) = length(get_connectivities(grid))
num_dofs(::ScalarField, grid::Grid) = num_allnodes(grid)
num_dofs(::VectorField, grid::Grid) = num_allnodes(grid) * get_dimension(grid)
num_elementdofs(::ScalarField, grid::Grid) = num_nodes(get_shape(grid))
num_elementdofs(::VectorField, grid::Grid) = num_nodes(get_shape(grid)) * get_dimension(grid)

dofindices(ftype::FieldType, grid::Grid, I) = dofindices(ftype, Val(get_dimension(grid)), I)

########################
# eachnode/eachelement #
########################

function eachnode(fieldtype::FieldType, grid::Grid)
    mappedarray(i -> dofindices(fieldtype, grid, i), get_nodeindices(grid))
end

function eachelement(fieldtype::FieldType, grid::Grid)
    mappedarray(conn -> dofindices(fieldtype, grid, conn), get_connectivities(grid))
end

#################
# generate_grid #
#################

default_shapetype(::Val{1}) = Line2()
default_shapetype(::Val{2}) = Quad4()
default_shapetype(::Val{3}) = Hex8()

# first order elements
function _connectivity(::Line2, I::CartesianIndex{1})
    CI = CartesianIndex
    i = I[1]
    (CI(i), CI(i+1))
end
function _connectivity(::Line3, I::CartesianIndex{1})
    CI = CartesianIndex
    i = I[1]
    (CI(i), CI(i+2), CI(i+1))
end
function _connectivity(::Quad4, I::CartesianIndex{2})
    CI = CartesianIndex
    i, j = I[1], I[2]
    (CI(i,j), CI(i+1,j), CI(i+1,j+1), CI(i,j+1))
end
function _connectivity(::Quad9, I::CartesianIndex{2})
    CI = CartesianIndex
    i, j = I[1], I[2]
    (CI(i,j), CI(i+2,j), CI(i+2,j+2), CI(i,j+2), CI(i+1,j), CI(i+2,j+1), CI(i+1,j+2), CI(i,j+1), CI(i+1,j+1))
end
function _connectivity(::Hex8, I::CartesianIndex{3})
    CI = CartesianIndex
    i, j, k = I[1], I[2], I[3]
    (CI(i,j,k), CI(i+1,j,k), CI(i+1,j+1,k), CI(i,j+1,k), CI(i,j,k+1), CI(i+1,j,k+1), CI(i+1,j+1,k+1), CI(i,j+1,k+1))
end
function _connectivity(::Hex27, I::CartesianIndex{3})
    CI = CartesianIndex
    i, j, k = I[1], I[2], I[3]
    (CI(i,j,k), CI(i+2,j,k), CI(i+2,j+2,k), CI(i,j+2,k), CI(i+1,j,k), CI(i+2,j+1,k), CI(i+1,j+2,k), CI(i,j+1,k), CI(i+1,j+1,k),
     CI(i,j,k+2), CI(i+2,j,k+2), CI(i+2,j+2,k+2), CI(i,j+2,k+2), CI(i+1,j,k+2), CI(i+2,j+1,k+2), CI(i+1,j+2,k+2), CI(i,j+1,k+2), CI(i+1,j+1,k+2),
     CI(i,j,k+1), CI(i+2,j,k+1), CI(i+2,j+2,k+1), CI(i,j+2,k+1), CI(i+1,j,k+1), CI(i+2,j+1,k+1), CI(i+1,j+2,k+1), CI(i,j+1,k+1), CI(i+1,j+1,k+1))
end

struct StructuredMesh{dim, Axes <: Tuple{Vararg{AbstractVector, dim}}} <: AbstractArray{Vec{dim, Float64}, dim}
    axes::Axes
    order::Int
end
Base.size(x::StructuredMesh) = x.order .* (map(length, x.axes) .- 1) .+ 1
primarysize(x::StructuredMesh) = map(length, x.axes)
function PrimaryCartesianIndices(x::StructuredMesh{dim}) where {dim}
    map(CartesianIndex{dim}, Iterators.product(map(ax -> first(ax):x.order:last(ax), axes(x))...))
end
@inline function _getindex(x::AbstractVector, i::Int, order::Int)::Float64
    @_propagate_inbounds_meta
    i_p, i_l = divrem(i-1, order) # primary and local index
    i_p += 1
    if i_l == 0
        x[i_p]
    else
        x[i_p] + i_l * (x[i_p+1] - x[i_p]) / order
    end
end
@inline function Base.getindex(x::StructuredMesh{dim}, I::Vararg{Int, dim}) where {dim}
    @_propagate_inbounds_meta
    Vec(broadcast(_getindex, x.axes, I, x.order))
end

# generate_grid
function generate_grid(::Type{T}, shape::Shape{dim}, axes::Vararg{AbstractVector, dim}) where {T, dim}
    mesh = StructuredMesh(axes, get_order(shape))
    nodes = vec(collect(Vec{dim, T}, mesh))
    primaryinds = PrimaryCartesianIndices(mesh)
    connectivities = map(CartesianIndices(size(primaryinds) .- 1)) do I
        Index{num_nodes(shape)}(broadcast(getindex, Ref(LinearIndices(mesh)), _connectivity(shape, primaryinds[I])))
    end
    Grid(shape, vec(nodes), vec(connectivities))
end
function generate_grid(::Type{T}, axes::Vararg{AbstractVector, dim}) where {T, dim}
    generate_grid(T, default_shapetype(Val(dim)), axes...)
end
function generate_grid(shape::Shape{dim}, axes::Vararg{AbstractVector, dim}) where {dim}
    Eltype = promote_type(map(eltype, axes)...)
    generate_grid(ifelse(Eltype==Int, Float64, Eltype), shape, axes...)
end
function generate_grid(axes::Vararg{AbstractVector, dim}) where {dim}
    Eltype = promote_type(map(eltype, axes)...)
    generate_grid(default_shapetype(Val(dim)), axes...)
end

########################
# integrate/integrate! #
########################

const MaybeTuple{T} = Union{T, Tuple{Vararg{T}}}

check_size(A::AbstractVector, n::Int) = @assert length(A) == n
check_size(A::AbstractMatrix, n::Int) = @assert size(A) == (n, n)

@pure function convert_integrate_function(f, eltindex)
    @inline function f′(qp, args...)
        @_propagate_inbounds_meta
        f(CartesianIndex(qp, eltindex), args...)
    end
end

## creating global matrix and vector
# matrix
function create_globalmatrix(::Type{T}, fieldtype::FieldType, grid::Grid) where {T}
    n = num_dofs(fieldtype, grid)
    sizehint = num_elementdofs(fieldtype, grid)^2 * num_elements(grid)
    SparseMatrixCOO{T}(n, n; sizehint)
end
create_globalmatrix(fieldtype::FieldType, grid::Grid{T}) where {T} = create_globalmatrix(T, fieldtype, grid)
# vector
function create_globalvector(::Type{T}, fieldtype::FieldType, grid::Grid) where {T}
    n = num_dofs(fieldtype, grid)
    zeros(T, n)
end
create_globalvector(fieldtype::FieldType, grid::Grid{T}) where {T} = create_globalvector(T, fieldtype, grid)
# from element array type
create_globalarray(::Type{T}, ::ElementArrayType{Matrix}, fieldtype::FieldType, grid::Grid) where {T} = create_globalmatrix(T, fieldtype, grid)
create_globalarray(::Type{T}, ::ElementArrayType{Vector}, fieldtype::FieldType, grid::Grid) where {T} = create_globalvector(T, fieldtype, grid)

## infer_integeltype
function infer_integeltype(f, arrtype::ElementArrayType, ftype::FieldType, grid::Grid)
    f′ = convert_integrate_function(f, 1) # use dummy eltindex = 1
    T = infer_integeltype(typeof(f′), typeof(arrtype), typeof(ftype), get_elementtype(grid))
    if T == Union{}
        first(build_element(f′, arrtype, ftype, create_element(grid), 1)) # run to throw error
        error("unreachable")
    end
    T
end

## integrate!
function integrate!(f::MaybeTuple{Function}, A::MaybeTuple{AbstractArray}, arrtype::MaybeTuple{ElementArrayType}, fieldtype::FieldType, grid::Grid; zeroinit::Bool = true)
    @~ check_size(A, num_dofs(fieldtype, grid))
    zeroinit && @~ fillzero!(A)
    element = create_element(grid)
    Ke = @~ create_elementarray((@~ eltype(A)), arrtype, fieldtype, element)
    for (eltindex, conn) in enumerate(get_connectivities(grid))
        update!(element, get_allnodes(grid)[conn])
        f′ = @~ convert_integrate_function(f, eltindex)
        @~ fillzero!(Ke)
        @~ integrate!(f′, Ke, arrtype, fieldtype, element)
        eltdofs = dofindices(fieldtype, grid, conn)
        @~ add!(A, eltdofs, Ke)
    end
    A
end
integrate!(f::MaybeTuple{Function}, A::MaybeTuple{AbstractArray}, fieldtype::FieldType, grid::Grid; zeroinit::Bool = true) =
    integrate!(f, A, (@~ ElementArrayType(f, get_elementtype(grid))), fieldtype, grid; zeroinit)

## integrate
@pure map_tupletype(f, ::Type{T}) where {T <: Tuple} = (map(f, T.parameters)...,)
@pure map_tupletype(f, ::Type{T}) where {T} = f(T)
function integrate(f::MaybeTuple{Function}, arrtype::MaybeTuple{ElementArrayType}, fieldtype::FieldType, grid::Grid)
    T = @~ infer_integeltype(f, arrtype, fieldtype, grid)
    A = @~ create_globalarray(T, arrtype, fieldtype, grid)
    integrate!(f, A, arrtype, fieldtype, grid)
end
integrate(f::MaybeTuple{Function}, fieldtype::FieldType, grid::Grid) =
    integrate(f, (@~ ElementArrayType(f, get_elementtype(grid))), fieldtype, grid)

#########################
# generate_elementstate #
#########################

function generate_elementstate(::Type{ElementState}, grid::Grid) where {ElementState}
    elementstate = StructArray{ElementState}(undef, num_quadpoints(get_shape(grid)), num_elements(grid))
    fillzero!(elementstate)
    if :x in propertynames(elementstate)
        elementstate.x .= interpolate(grid, get_allnodes(grid))
    end
    elementstate
end

###############
# interpolate #
###############

# returned mappedarray's size is the same as elementstate matrix
function interpolate(grid::Grid{T}, nodalvalues::AbstractVector) where {T}
    @assert num_allnodes(grid) == length(nodalvalues)
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
