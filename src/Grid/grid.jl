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

get_order(::SingleField{nothing}, grid::Grid) = get_order(get_shape(grid))
get_order(field::SingleField, grid::Grid) = get_order(field)
get_order(mixed::MixedField, grid::Grid) = maximum(fld -> get_order(fld, grid), mixed.fields)

get_allnodes(grid::Grid) = grid.nodes
function get_allnodes(grid::Grid, order::Int)
    nodes = get_allnodes(grid)
    inds = Block.(1:order)
    @boundscheck checkbounds(Bool, nodes, inds) || error("order $order is not available in Grid with $(get_shape(grid))")
    @inbounds view(nodes, inds)
end
get_allnodes(field::Field, grid::Grid) = get_allnodes(grid, get_order(field, grid))
function get_allnodes_flatten(grid::Grid)
    nodes = get_allnodes(grid)
    reinterpret(reshape, eltype(eltype(nodes)), nodes)
end

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
    field = argmax(fld -> get_order(fld, grid), mixed.fields)
    get_connectivity(field, grid, i)
end

get_dimension(grid::Grid{<: Any, dim}) where {dim} = dim

get_shape(grid::Grid) = grid.shape
get_shape(field::Field, grid::Grid) = get_available_shapes(grid)[get_order(field, grid)]
get_available_shapes(grid::Grid) = get_lower_shapes(get_shape(grid))

create_element(grid::Grid{T, dim}) where {T, dim} = Element{T, dim}(get_shape(grid))
create_element(field::SingleField, grid::Grid{T, dim}) where {T, dim} = Element{T, dim}(get_shape(field, grid))
create_element(mixed::MixedField, grid::Grid{T, dim}) where {T, dim} = Element{T, dim}(map(field -> get_shape(field, grid), mixed.fields))
get_elementtype(field::Field, grid::Grid) = Base._return_type(create_element, typeof((field, grid)))

num_allnodes(grid::Grid) = length(get_allnodes(grid))
function num_allnodes(field::Field, grid::Grid)
    order = get_order(field, grid)
    nodes = get_allnodes(grid)
    n = 0
    @inbounds @simd for i in 1:order
        n += length(nodes[Block(i)])
    end
    n
end
num_elements(grid::Grid) = length(get_connectivities(grid))

num_dofs(field::ScalarField, grid::Grid) = num_allnodes(field, grid)
num_dofs(field::VectorField, grid::Grid) = num_allnodes(field, grid) * get_dimension(grid)
num_dofs(mixed::MixedField, grid::Grid) = sum(field -> num_dofs(field, grid), mixed.fields)

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
        map(mixed.fields) do field
            offset = count[]
            count[] += num_dofs(field, grid)
            $func(field, grid; offset)
        end
    end
end

function get_elementdofs(field::SingleField, grid::Grid, i::Int; offset::Int = 0)
    dofindices(field, Val(get_dimension(grid)), get_connectivity(field, grid, i)) .+ offset
end
@inline function get_elementdofs(mixed::MixedField, grid::Grid, i::Int)
    count = Ref(0)
    reduce(vcat, map(mixed.fields) do field
        @_inline_meta
        offset = count[]
        count[] += num_dofs(field, grid)
        get_elementdofs(field, grid, i; offset)
    end)
end

###############
# interpolate #
###############

# returned mappedarray's size is the same as elementstate matrix
function interpolate(field::Field, grid::Grid, nodalvalues::AbstractVector)
    @assert num_dofs(field, grid) == length(nodalvalues)
    element = create_element(field, grid)
    dims = (num_quadpoints(element), num_elements(grid))
    last_eltindex = Ref(0)
    mappedarray(CartesianIndices(dims)) do I
        qp, eltindex = Tuple(I)
        if last_eltindex[] != eltindex
            conn = get_connectivity(field, grid, eltindex)
            update!(element, get_allnodes(grid)[conn])
            last_eltindex[] = eltindex
        end
        dofs = get_elementdofs(field, grid, eltindex)
        interpolate(field, element, nodalvalues[dofs], qp)
    end
end

#############
# integrate #
#############

@pure function convert_integrate_function(f, eltindex)
    @inline function f′(qp, args...)
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
    @eval function integrate!(f, A::$ArrayType, field::Field, grid::Grid; zeroinit::Bool=true, kwargs...)
        Ke = $create_array(eltype(A), field, create_element(field, grid))
        integrate_lowered!(A, field, grid; zeroinit) do eltindex, element
            f′ = convert_integrate_function(f, eltindex)
            fillzero!(Ke)
            integrate!(f′, Ke, field, element; kwargs...)
            Ke
        end
    end
end

# integrate
function integrate(f, field::Field, grid::Grid; kwargs...)
    F = integrate_function(f, get_elementtype(field, grid))
    F(f, field, grid; kwargs...)
end

# special version for AD

function integrate!(f, F::AbstractVector, K::AbstractMatrix, field::Field, grid::Grid, U::AbstractVector; zeroinit::Tuple{Bool, Bool} = (true, true))
    @assert num_dofs(field, grid) == length(U)
    zeroinit[1] && fillzero!(F)
    zeroinit[2] && fillzero!(K)
    element = create_element(field, grid)
    Fe = create_vector(eltype(F), field, element)
    Ke = create_matrix(eltype(K), field, element)
    for eltindex in 1:num_elements(grid)
        # update element
        conn = get_connectivity(field, grid, eltindex)
        update!(element, get_allnodes(grid)[conn])
        # integration in an element
        f′ = convert_integrate_function(f, eltindex)
        fillzero!(Fe)
        fillzero!(Ke)
        dofs = get_elementdofs(field, grid, eltindex)
        integrate!(f′, Fe, Ke, field, element, U[dofs])
        add!(F, dofs, Fe)
        add!(K, dofs, Ke)
    end
    F, K
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

function gridvalues(U::AbstractVector, mixed::MixedField, grid::Grid)
    map(mixed.fields, get_dofs(mixed, grid)) do field, dofs
        gridvalues(view(U, dofs), field, grid)
    end
end

####################
# sparsity_pattern #
####################

function sparsity_pattern(field::Field, grid::Grid{T}) where {T}
    sparse(integrate((i,v,u)->zero(T), field, grid))
end
