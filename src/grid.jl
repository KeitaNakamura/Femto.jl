# shape must be unique in grid
struct Grid{T, dim, shape_dim, S <: Shape{shape_dim}, L}
    nodes::Vector{Vec{dim, T}}
    nodeindices::Vector{Int}
    shape::S
    connectivities::Vector{Index{L}}
end

get_nodes(grid::Grid) = grid.nodes
get_nodeindices(grid::Grid) = grid.nodeindices
get_shape(grid::Grid) = grid.shape
get_connectivities(grid::Grid) = grid.connectivities
get_dimension(grid::Grid{<: Any, dim}) where {dim} = dim
get_elementtype(grid::Grid) = Base._return_type(create_element, Tuple{typeof(grid)})
create_element(grid::Grid{T, dim}) where {T, dim} = Element{T, dim}(get_shape(grid))

num_nodes(grid::Grid) = length(get_nodes(grid))
num_elements(grid::Grid) = length(get_connectivities(grid))
num_dofs(::ScalarField, grid::Grid) = num_nodes(grid)
num_dofs(::VectorField, grid::Grid) = num_nodes(grid) * get_dimension(grid)
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
    Grid(vec(nodes), collect(1:length(nodes)), _shapetype(Val(dim)), vec(connectivities))
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
    sizehint = 2*num_elementdofs(fieldtype, grid) * num_elements(grid)
    SparseMatrixIJV{T}(n, n; sizehint)
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
        update!(element, get_nodes(grid)[conn])
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
