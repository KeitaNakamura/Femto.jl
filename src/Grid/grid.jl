const BVector{T} = BlockVector{T, Vector{Vector{T}}, Tuple{BlockedUnitRange{Vector{Int}}}}
_mortar(x::BVector) = x
_mortar(x::Vector) = mortar([x])

struct Grid{T, dim, shape_dim, S <: Shape{shape_dim}, L}
    # unique in gridset
    nodes::BVector{Vec{dim, T}}
    # body/face element
    shape::S
    connectivities::Vector{Index{L}}
    nodeindices::BVector{Int}
    function Grid(nodes::AbstractVector{Vec{dim, T}}, shape::S, conns::Vector{Index{L}}, inds::AbstractVector{Int} = collect(1:length(nodes))) where {T, dim, shape_dim, S <: Shape{shape_dim}, L}
        @assert num_nodes(shape) == L
        new{T, dim, shape_dim, S, L}(_mortar(nodes), shape, conns, _mortar(inds))
    end
end

function decrease_order(grid::Grid)
    shape = decrease_order(get_shape(grid))
    order = get_order(shape)
    nodes = collect(get_allnodes(grid, order))
    conns = map(get_connectivities(grid)) do conn
        conn[SOneTo(num_nodes(shape))]
    end
    inds = collect(get_nodeindices(grid, order))
    Grid(nodes, shape, conns, inds)
end

#########
# utils #
#########

get_order(::SingleFieldNoOrder, grid::Grid) = get_order(get_shape(grid))
get_order(field::SingleField, grid::Grid) = get_order(field)
get_order(mixed::MixedField, grid::Grid) = maximum(fld -> get_order(fld, grid), get_singlefields(mixed))

get_allnodes(grid::Grid) = grid.nodes
function get_allnodes(grid::Grid, order::Int)
    nodes = get_allnodes(grid)
    inds = Block.(1:order)
    @boundscheck checkbounds(Bool, nodes, inds) || error("order $order is not available in Grid with $(get_shape(grid))")
    @inbounds view(nodes, inds)
end
get_allnodes(field::Field, grid::Grid) = get_allnodes(grid, get_order(field, grid))

get_nodeindices(grid::Grid) = grid.nodeindices
function get_nodeindices(grid::Grid, order::Int)
    nodeinds = get_nodeindices(grid)
    inds = Block.(1:order)
    @boundscheck checkbounds(Bool, nodeinds, inds) || error("order $order is not available in Grid with $(get_shape(grid))")
    @inbounds view(nodeinds, inds)
end
get_nodeindices(field::Field, grid::Grid) = get_nodeindices(grid, get_order(field, grid))

get_connectivities(grid::Grid) = grid.connectivities
function _get_connectivity(shape::Shape, grid::Grid, i::Int)
    conns = get_connectivities(grid)
    @boundscheck checkbounds(conns, i)
    @inbounds conn = conns[i]
    conn[SOneTo(num_nodes(shape))]
end
function get_connectivity(grid::Grid, i::Int)
    @_propagate_inbounds_meta
    _get_connectivity(get_shape(grid), grid, i)
end
function get_connectivity(field::SingleField, grid::Grid, i::Int)
    @_propagate_inbounds_meta
    _get_connectivity(get_shape(field, grid), grid, i)
end
function get_connectivity(mixed::MixedField, grid::Grid, i::Int)
    field = argmax(fld -> get_order(fld, grid), get_singlefields(mixed))
    get_connectivity(field, grid, i)
end

get_dimension(grid::Grid{<: Any, dim}) where {dim} = dim

get_shape(grid::Grid) = grid.shape
get_shape(field::Field, grid::Grid) = get_available_shapes(grid)[get_order(field, grid)]
get_available_shapes(grid::Grid) = get_lower_shapes(get_shape(grid))

create_element(grid::Grid{T, dim}) where {T, dim} = Element{T, dim}(get_shape(grid))
create_element(field::SingleField, grid::Grid{T, dim}) where {T, dim} = Element{T, dim}(get_shape(field, grid))
create_element(mixed::MixedField{N}, grid::Grid{T, dim}) where {N, T, dim} = Element{T, dim}(ntuple(i -> get_shape(get_singlefields(mixed)[i], grid), Val(N)))
get_elementtype(field::Field, grid::Grid) = Base._return_type(create_element, typeof((field, grid)))

num_allnodes(grid::Grid) = length(get_allnodes(grid))
num_allnodes(field::Field, grid::Grid) = length(get_allnodes(field, grid))
num_elements(grid::Grid) = length(get_connectivities(grid))

num_dofs(field::ScalarField, grid::Grid) = num_allnodes(field, grid)
num_dofs(field::VectorField, grid::Grid) = num_allnodes(field, grid) * get_dimension(grid)
num_dofs(mixed::MixedField, grid::Grid) = sum(field -> num_dofs(field, grid), get_singlefields(mixed))

########
# dofs #
########

get_dofs(field::SingleField, grid::Grid; offset::Int = 0) = convert(Vector{Int}, get_nodedofs(field, grid; offset))
get_dofs(field::VectorField, grid::Grid; offset::Int = 0) = reduce(vcat, get_nodedofs(field, grid; offset))

function get_nodedofs(field::SingleField, grid::Grid; offset::Int = 0)
    nodeinds = get_nodeindices(field, grid)
    map(i -> dofindices(field, Val(get_dimension(grid)), i) .+ offset, nodeinds)
end

for func in (:get_dofs, :get_nodedofs)
    @eval function $func(mixed::MixedField, grid::Grid)
        count = Ref(0)
        map(get_singlefields(mixed)) do field
            offset = count[]
            count[] += num_dofs(field, grid)
            $func(field, grid; offset)
        end
    end
end

function get_elementdofs(field::SingleField, grid::Grid, i::Int; offset::Int = 0)
    dofindices(field, Val(get_dimension(grid)), get_connectivity(field, grid, i)) .+ offset
end
function get_elementdofs(mixed::MixedField, grid::Grid, i::Int)
    count = Ref(0)
    reduce(vcat, map(get_singlefields(mixed)) do field
        offset = count[]
        count[] += num_dofs(field, grid)
        get_elementdofs(field, grid, i; offset)
    end)
end

###############
# interpolate #
###############

# returned mappedarray's size is the same as elementstate matrix
function interpolate(grid::Grid, nodalvalues::AbstractVector)
    @assert num_allnodes(grid) == length(nodalvalues)
    element = create_element(grid)
    dims = (num_quadpoints(element), num_elements(grid))
    mappedarray(CartesianIndices(dims)) do I
        qp, eltindex = Tuple(I)
        conn = get_connectivities(grid)[eltindex]
        update!(element, get_allnodes(grid)[conn])
        interpolate(element, nodalvalues[conn], qp)
    end
end

interpolate(::ScalarField, grid::Grid, nodalvalues::AbstractVector{<: Real}) = interpolate(grid, nodalvalues)
interpolate(::VectorField, grid::Grid{T, dim}, nodalvalues::AbstractVector{<: Real}) where {T, dim} = interpolate(grid, reinterpret(Vec{dim, T}, nodalvalues))

#############
# integrate #
#############

@pure function convert_integrate_function(f, eltindex)
    @inline function f′(qp, args...)
        @_propagate_inbounds_meta
        f(CartesianIndex(qp, eltindex), args...)
    end
end

function create_matrix(::Type{T}, field::Field, grid::Grid) where {T}
    m = n = num_dofs(field, grid)
    element = create_element(field, grid)
    sizehint = num_dofs(field, element)^2 * num_elements(grid)
    SparseMatrixCOO{T}(m, n; sizehint)
end
function create_vector(::Type{T}, field::Field, grid::Grid) where {T}
    n = num_dofs(field, grid)
    zeros(T, n)
end

## infer
for infer in (:infer_integrate_matrix_eltype, :infer_integrate_vector_eltype)
    _infer = Symbol(:_, infer)
    @eval function $infer(f, field::Field, grid::Grid)
        f′ = convert_integrate_function(f, 1) # use dummy eltindex = 1
        $_infer(f′, typeof(field), get_elementtype(field, grid))
    end
end

## integrate_lowered!
function integrate_lowered!(f, A::AbstractArray, field::Field, grid::Grid; zeroinit::Bool = true)
    @assert all(==(num_dofs(field, grid)), size(A))
    zeroinit && fillzero!(A)
    element = create_element(field, grid)
    for eltindex in 1:num_elements(grid)
        conn = get_connectivity(field, grid, eltindex)
        update!(element, get_allnodes(grid)[conn])
        Ke = f(eltindex, element)
        dofs = get_elementdofs(field, grid, eltindex)
        add!(A, dofs, Ke)
    end
    A
end

## integrate!
for (ArrayType, create_array) in ((:AbstractMatrix, :create_matrix), (:AbstractVector, :create_vector))
    @eval function integrate!(f, A::$ArrayType, field::Field, grid::Grid; zeroinit::Bool = true)
        Ke = $create_array(eltype(A), field, create_element(field, grid))
        integrate_lowered!(A, field, grid; zeroinit) do eltindex, element
            f′ = convert_integrate_function(f, eltindex)
            fillzero!(Ke)
            integrate!(f′, Ke, field, element)
            Ke
        end
    end
end

# integrate
function integrate(f, field::Field, grid::Grid)
    F = integrate_function(f, get_elementtype(field, grid))
    F(f, field, grid)
end

##############
# gridvalues #
##############

grideltype(::Type{T}, ::ScalarField, ::Grid) where {T <: Real} = T
grideltype(::Type{T}, ::VectorField, grid::Grid) where {T <: Real} = Vec{get_dimension(grid), T}

function gridvalues(U::AbstractVector, field::SingleField, grid::Grid)
    n = num_dofs(field, grid)
    @assert length(U) == n
    ElType = grideltype(eltype(U), field, grid)
    reinterpret(ElType, U)
end

function gridvalues(U::AbstractVector, mixed::MixedField{N}, grid::Grid) where {N}
    map(get_singlefields(mixed), get_dofs(mixed, grid)) do field, dofs
        gridvalues(view(U, dofs), field, grid)
    end
end
