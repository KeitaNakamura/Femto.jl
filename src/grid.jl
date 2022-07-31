struct DomainInfo{S <: Shape, L}
    shape::S
    connectivities::Vector{Index{L}}
    nodeindices::Vector{Int}
end

function generate_gridset(nodes::Vector{Vec{dim, T}}, domains::Dict{String, DomainInfo}) where {dim, T}
    bodies = Dict(Iterators.filter(p->get_dimension(p.second.shape)==dim, domains))
    faces  = Dict(Iterators.filter(p->get_dimension(p.second.shape)!=dim, domains))
    perm = get_perm_lower_shapes(
        nodes,
        only(unique(Iterators.map(p->p.second.shape, bodies))),
        reduce(vcat, Iterators.map(p->p.second.connectivities, bodies)),
    )
    new_nodes = permute_nodes(nodes, perm)
    Dict{String, Grid{T, dim}}(Iterators.map(domains) do (name, domain)
        connectivities, nodeindices = permute_connectivities_nodeindices(domain.connectivities, domain.nodeindices, perm)
        name => Grid(new_nodes, domain.shape, connectivities, nodeindices)
    end)
end

function get_perm_lower_shapes(nodes::Vector{<: Vec}, shape::Shape, connectivities::Vector{Index{L}}) where {L}
    @assert num_nodes(shape) == L
    count = Ref(1)
    shapes = get_lower_shapes(shape)
    shapes === () && return nothing
    perm = mortar(collect(map(shapes) do s
        start = count[]
        stop = num_nodes(s)
        count[] += stop
        unique!(sort!(reduce(vcat, [view(conn, start:stop) for conn in connectivities])))
    end))
    @assert length(nodes) == length(perm)
    @assert allunique(perm)
    perm
end

function permute_nodes(nodes::Vector{<: Vec}, perm::BlockVector{Int})
    mortar(map(eachblock(perm)) do p
        nodes[p]
    end)
end
permute_nodes(nodes::Vector{<: Vec}, perm::Nothing) = nodes

function permute_connectivities_nodeindices(connectivities::Vector{Index{L}}, nodeindices::Vector{Int}, perm::BlockVector{Int}) where {L}
    perm_inv = invperm(perm)
    new_connectivities = Index{L}[perm_inv[conn] for conn in connectivities]
    new_nodeindices = perm_inv[nodeindices]
    sort!(new_nodeindices)
    count = Ref(0)
    upper = Ref(0)
    blocked_nodeindices = mortar(map(eachblock(perm)) do p
        upper[] += length(p)
        offset = count[]
        stop = findlast(≤(upper[]), new_nodeindices)
        count[] = stop
        new_nodeindices[offset+1:stop]
    end)
    new_connectivities, blocked_nodeindices
end
permute_connectivities_nodeindices(connectivities::Vector{<: Index}, nodeindices::Vector{Int}, perm::Nothing) = (connectivities, nodeindices)

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
    function Grid(nodes::AbstractVector{Vec{dim, T}}, shape::S, connectivities::Vector{Index{L}}, nodeindices::AbstractVector{Int} = collect(1:length(nodes))) where {T, dim, shape_dim, S <: Shape{shape_dim}, L}
        @assert num_nodes(shape) == L
        new{T, dim, shape_dim, S, L}(_mortar(nodes), shape, connectivities, _mortar(nodeindices))
    end
end

function decrease_order(grid::Grid)
    shape = decrease_order(get_shape(grid))
    order = get_order(shape)
    conns = map(get_connectivities(grid)) do conn
        conn[SOneTo(num_nodes(shape))]
    end
    Grid(collect(get_allnodes(grid, order)), shape, conns, collect(get_nodeindices(grid, order)))
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

function get_dofs(field::SingleField, grid::Grid; offset::Int = 0)
    convert(Vector{Int}, get_nodedofs(field, grid; offset))
end
function get_dofs(field::VectorField, grid::Grid; offset::Int = 0)
    reduce(vcat, get_nodedofs(field, grid; offset))
end

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
    (CI(i,j,k), CI(i+2,j,k), CI(i+2,j+2,k), CI(i,j+2,k), CI(i,j,k+2), CI(i+2,j,k+2), CI(i+2,j+2,k+2), CI(i,j+2,k+2),
     CI(i+1,j,k), CI(i,j+1,k), CI(i,j,k+1), CI(i+2,j+1,k), CI(i+2,j,k+1), CI(i+1,j+2,k), CI(i+2,j+2,k+1), CI(i,j+2,k+1), CI(i+1,j,k+2), CI(i,j+1,k+2), CI(i+2,j+1,k+2), CI(i+1,j+2,k+2),
     CI(i+1,j+1,k), CI(i+1,j,k+1), CI(i,j+1,k+1), CI(i+2,j+1,k+1), CI(i+1,j+2,k+1), CI(i+1,j+1,k+2), CI(i+1,j+1,k+1))
end
function generate_connectivities(shape::Shape, nodeindices::AbstractArray{Int})
    primaryinds = primaryindices(size(nodeindices), get_order(shape))
    map(CartesianIndices(size(primaryinds) .- 1)) do I
        Index(broadcast(getindex, Ref(nodeindices), _connectivity(shape, primaryinds[I])))
    end |> vec
end

# helper mesh type
struct StructuredMesh{dim, Axes <: Tuple{Vararg{AbstractVector, dim}}} <: AbstractArray{Vec{dim, Float64}, dim}
    axes::Axes
    order::Int
end
Base.size(x::StructuredMesh) = x.order .* (map(length, x.axes) .- 1) .+ 1
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
function primaryindices(dims::Dims{dim}, order::Int) where {dim}
    map(CartesianIndex{dim}, Iterators.product(map(l -> 1:order:l, dims)...))
end

# generate_grid
grid_args(::Type{T}, shape::Shape, axes::AbstractVector...) where {T} = (T, shape, axes...)
grid_args(::Type{T}, axes::Vararg{AbstractVector, dim}) where {T, dim} = grid_args(T, default_shapetype(Val(dim)), axes...)
grid_args(shape::Shape{dim}, axes::Vararg{AbstractVector, dim}) where {dim} = (T = promote_type(map(eltype, axes)...); grid_args(ifelse(T==Int, Float64, T), shape, axes...))
grid_args(axes::Vararg{AbstractVector, dim}) where {dim} = grid_args(default_shapetype(Val(dim)), axes...)

generate_grid(args...) = _generate_grid(grid_args(args...)...)
function _generate_grid(::Type{T}, shape::Shape{shape_dim}, axes::Vararg{AbstractVector, dim}) where {T, dim, shape_dim}
    mesh = StructuredMesh(axes, get_order(shape))
    nodes = vec(collect(Vec{dim, T}, mesh))
    connectivities = generate_connectivities(shape, LinearIndices(mesh))
    perm = get_perm_lower_shapes(nodes, shape, connectivities)
    Grid(
        permute_nodes(nodes, perm),
        shape,
        permute_connectivities_nodeindices(connectivities, collect(1:length(nodes)), perm)...,
    )
end

####################
# generate_gridset #
####################

boundshape(::Quad4) = Line2()
boundshape(::Quad9) = Line3()
boundshape(::Hex8) = Quad4()
boundshape(::Hex27) = Quad9()

generate_gridset(args...) = _generate_gridset(grid_args(args...)...)
function _generate_gridset(::Type{T}, shape::Shape{2}, axes::Vararg{AbstractVector, 2}) where {T}
    bshape = boundshape(shape)
    order = get_order(shape)
    mesh = StructuredMesh(axes, order)
    nodeinds = LinearIndices(mesh)

    # nodeinds
    nodeinds_left = nodeinds[1,end:-1:1]
    nodeinds_right = nodeinds[end,1:end]
    nodeinds_bottom = nodeinds[1:end,1]
    nodeinds_top = nodeinds[end:-1:1,end]
    nodeinds_boundary = [nodeinds_left; nodeinds_right; nodeinds_bottom; nodeinds_top]

    # connectivities
    main = generate_connectivities(shape, nodeinds)
    left = generate_connectivities(bshape, nodeinds_left)
    right = generate_connectivities(bshape, nodeinds_right)
    bottom = generate_connectivities(bshape, nodeinds_bottom)
    top = generate_connectivities(bshape, nodeinds_top)
    boundary = [left; right; bottom; top]
    boundary = mapreduce(vec, vcat, (left, right, bottom, top))

    allnodes = vec(collect(Vec{2, T}, mesh))
    domains = Dict{String, DomainInfo}(
        "main"   => DomainInfo(shape, main, collect(1:length(allnodes))),
        "left"   => DomainInfo(bshape, left, sort!(nodeinds_left)),
        "right"  => DomainInfo(bshape, right, sort!(nodeinds_right)),
        "bottom" => DomainInfo(bshape, bottom, sort!(nodeinds_bottom)),
        "top"    => DomainInfo(bshape, top, sort!(nodeinds_top)),
        "boundary" => DomainInfo(bshape, boundary, unique!(sort!(nodeinds_boundary)))
    )
    generate_gridset(allnodes, domains)
end
function _generate_gridset(::Type{T}, shape::Shape{3}, axes::Vararg{AbstractVector, 3}) where {T}
    bshape = boundshape(shape)
    mesh = StructuredMesh(axes, get_order(shape))
    nodeinds = LinearIndices(mesh)

    # nodeinds
    nodeinds_left   = nodeinds[1,        1:end, 1:end   ] |> permutedims
    nodeinds_right  = nodeinds[end,      1:end, end:-1:1] |> permutedims
    nodeinds_bottom = nodeinds[1:end,    1,     1:end   ]
    nodeinds_top    = nodeinds[1:end,    end,   end:-1:1]
    nodeinds_front  = nodeinds[1:end,    1:end, end     ]
    nodeinds_back   = nodeinds[end:-1:1, 1:end, 1       ]
    nodeinds_boundary = mapreduce(vec, vcat, (nodeinds_left, nodeinds_right, nodeinds_bottom, nodeinds_top, nodeinds_front, nodeinds_back))

    # connectivities
    main = generate_connectivities(shape, nodeinds)
    left = generate_connectivities(bshape, nodeinds_left)
    right = generate_connectivities(bshape, nodeinds_right)
    bottom = generate_connectivities(bshape, nodeinds_bottom)
    top = generate_connectivities(bshape, nodeinds_top)
    front = generate_connectivities(bshape, nodeinds_front)
    back = generate_connectivities(bshape, nodeinds_back)
    boundary = [left; right; bottom; top; front; back]

    allnodes = vec(collect(Vec{3, T}, mesh))
    domains = Dict{String, DomainInfo}(
        "main"   => DomainInfo(shape, main, collect(1:length(allnodes))),
        "left"   => DomainInfo(bshape, left, sort!(vec(nodeinds_left))),
        "right"  => DomainInfo(bshape, right, sort!(vec(nodeinds_right))),
        "bottom" => DomainInfo(bshape, bottom, sort!(vec(nodeinds_bottom))),
        "top"    => DomainInfo(bshape, top, sort!(vec(nodeinds_top))),
        "front"  => DomainInfo(bshape, front, sort!(vec(nodeinds_front))),
        "back"   => DomainInfo(bshape, back, sort!(vec(nodeinds_back))),
        "boundary" => DomainInfo(bshape, boundary, unique!(sort!(vec(nodeinds_boundary))))
    )
    generate_gridset(allnodes, domains)
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
function infer_integrate_matrix_eltype(f, field::Field, grid::Grid)
    f′ = convert_integrate_function(f, 1) # use dummy eltindex = 1
    _infer_integrate_matrix_eltype(f′, typeof(field), get_elementtype(field, grid))
end
function infer_integrate_vector_eltype(f, field::Field, grid::Grid)
    f′ = convert_integrate_function(f, 1) # use dummy eltindex = 1
    _infer_integrate_vector_eltype(f′, typeof(field), get_elementtype(field, grid))
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

#########################
# generate_elementstate #
#########################

function generate_elementstate(::Type{ElementState}, grid::Grid) where {ElementState}
    shape = get_shape(grid)
    elementstate = StructArray{ElementState}(undef, num_quadpoints(shape), num_elements(grid))
    fillzero!(elementstate)
    if :x in propertynames(elementstate)
        elementstate.x .= interpolate(grid, get_allnodes(grid))
    end
    elementstate
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
