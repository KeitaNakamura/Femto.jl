struct DomainInfo{S <: Shape, L}
    shape::S
    connectivities::Vector{Index{L}}
    nodeindices::Vector{Int}
end

function generate_gridset(nodes::Vector{Vec{dim, T}}, domains::Dict) where {dim, T}
    grids = generate_gridset(nodes, values(domains))
    Dict([name=>grid for (name, grid) in zip(keys(domains), grids)])
end
function generate_gridset(nodes::Vector{Vec{dim, T}}, domains) where {dim, T}
    bodies = Iterators.filter(dm->get_dimension(dm.shape)==dim, domains)
    faces  = Iterators.filter(dm->get_dimension(dm.shape)!=dim, domains)
    perm = get_perm_lower_shapes(
        nodes,
        only(unique(Iterators.map(b->b.shape, bodies))),
        reduce(vcat, Iterators.map(b->b.connectivities, bodies)),
    )
    new_nodes = permute_nodes(nodes, perm)
    map(domains) do domain
        connectivities, nodeindices = permute_connectivities_nodeindices(domain.connectivities, domain.nodeindices, perm)
        Grid(new_nodes, domain.shape, connectivities, nodeindices)
    end
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
function _connectivity(::Line4, I::CartesianIndex{1})
    CI = CartesianIndex
    i = I[1]
    (CI(i), CI(i+3), CI(i+1), CI(i+2))
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
function generate_connectivities(shape::Shape, nodeindices::AbstractArray{Int})::Vector{Index{num_nodes(shape)}}
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
grid_args(::Type{T}, shape::Shape, axes::AbstractVector{<: Real}...) where {T} = (T, shape, axes...)
grid_args(::Type{T}, axes::Vararg{AbstractVector{<: Real}, dim}) where {T, dim} = grid_args(T, default_shapetype(Val(dim)), axes...)
grid_args(shape::Shape{dim}, axes::Vararg{AbstractVector{<: Real}, dim}) where {dim} = (T = promote_type(map(eltype, axes)...); grid_args(ifelse(T==Int, Float64, T), shape, axes...))
grid_args(axes::Vararg{AbstractVector{<: Real}, dim}) where {dim} = grid_args(default_shapetype(Val(dim)), axes...)

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

#########################
# generate_elementstate #
#########################

function generate_elementstate(::Type{ElementState}, grid::Grid{T}) where {ElementState, T}
    shape = get_shape(grid)
    elementstate = StructArray{ElementState}(undef, num_quadpoints(shape), num_elements(grid))
    fillzero!(elementstate)
    if :x in propertynames(elementstate)
        elementstate.x .= interpolate(VectorField(), grid, reinterpret(T, get_allnodes(grid)))
    end
    elementstate
end
