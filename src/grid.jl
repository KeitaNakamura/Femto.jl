struct Grid{T, dim, Elt <: Element{T, dim}, L}
    # unique in gridset
    nodes::Vector{Vec{dim, T}}
    # body/face element
    element::Elt
    connectivities::Vector{Index{L}}
    nodeindices::Vector{Int}
end

function Grid(nodes::Vector{Vec{dim, T}}, shape::Shape, connectivities::Vector{Index{L}}, nodeindices::Vector{Int} = collect(1:length(nodes))) where {dim, T, L}
    @assert num_nodes(shape) ≤ L
    element = Element{T, dim}(shape)
    Grid(nodes, element, connectivities, nodeindices)
end

# core fields in grid are inherited
function Grid(grid::Grid, shape::Shape, connectivities::Vector{<: Index}, nodeindices::Vector{Int})
    Grid(get_allnodes(grid), shape, connectivities, nodeindices)
end

#########
# utils #
#########

get_allnodes(grid::Grid) = grid.nodes
get_nodeindices(grid::Grid) = grid.nodeindices
get_connectivities(grid::Grid) = grid.connectivities
get_dimension(grid::Grid{<: Any, dim}) where {dim} = dim
get_element(grid::Grid) = grid.element
function get_element(grid::Grid, i::Int)
    @boundscheck 1 ≤ i ≤ num_elements(grid)
    @inbounds conn = get_connectivities(grid)[i]
    element = get_element(grid)
    update!(element, get_allnodes(grid)[conn])
    element
end
get_shape(grid::Grid) = get_shape(get_element(grid))

num_allnodes(grid::Grid) = length(get_allnodes(grid))
num_elements(grid::Grid) = length(get_connectivities(grid))
num_dofs(::ScalarField, grid::Grid) = num_allnodes(grid)
num_dofs(::VectorField, grid::Grid) = num_allnodes(grid) * get_dimension(grid)

########
# dofs #
########

function nodedofs(field::SingleField, grid::Grid)
    mappedarray(i -> dofindices(field, Val(get_dimension(grid)), i), get_nodeindices(grid))
end

function elementdofs(field::SingleField, grid::Grid, i::Int)
    conns = get_connectivities(grid)
    @boundscheck checkbounds(conns, i)
    @inbounds conn = conns[i]
    dofindices(field, Val(get_dimension(grid)), conn)
end
function elementdofs(field::SingleField, grid::Grid)
    mappedarray(i -> elementdofs(field, grid, i), 1:num_elements(grid))
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
    Grid(nodes, shape, connectivities)
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
    grid = Grid(allnodes, shape, main)
    Dict{String, Grid{T, 2}}(
        "main"   => grid,
        "left"   => Grid(grid, bshape, left, sort!(nodeinds_left)),
        "right"  => Grid(grid, bshape, right, sort!(nodeinds_right)),
        "bottom" => Grid(grid, bshape, bottom, sort!(nodeinds_bottom)),
        "top"    => Grid(grid, bshape, top, sort!(nodeinds_top)),
        "boundary" => Grid(grid, bshape, boundary, sort!(nodeinds_boundary))
    )
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
    grid = Grid(allnodes, shape, main)
    Dict{String, Grid{T, 3}}(
        "main"   => grid,
        "left"   => Grid(grid, bshape, left, sort!(vec(nodeinds_left))),
        "right"  => Grid(grid, bshape, right, sort!(vec(nodeinds_right))),
        "bottom" => Grid(grid, bshape, bottom, sort!(vec(nodeinds_bottom))),
        "top"    => Grid(grid, bshape, top, sort!(vec(nodeinds_top))),
        "front"  => Grid(grid, bshape, front, sort!(vec(nodeinds_front))),
        "back"   => Grid(grid, bshape, back, sort!(vec(nodeinds_back))),
        "boundary" => Grid(grid, bshape, boundary, sort!(vec(nodeinds_boundary)))
    )
end

###############
# interpolate #
###############

# returned mappedarray's size is the same as elementstate matrix
function interpolate(grid::Grid, nodalvalues::AbstractVector)
    @assert num_allnodes(grid) == length(nodalvalues)
    dims = (num_quadpoints(get_element(grid)), num_elements(grid))
    mappedarray(CartesianIndices(dims)) do I
        qp, eltindex = Tuple(I)
        conn = get_connectivities(grid)[eltindex]
        element = get_element(grid, eltindex)
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
    element = get_element(grid)
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
    _infer_integrate_matrix_eltype(f′, typeof(field), typeof(get_element(grid)))
end
function infer_integrate_vector_eltype(f, field::Field, grid::Grid)
    f′ = convert_integrate_function(f, 1) # use dummy eltindex = 1
    _infer_integrate_vector_eltype(f′, typeof(field), typeof(get_element(grid)))
end

## integrate_lowered!
function integrate_lowered!(f, A::AbstractArray, field::Field, grid::Grid; zeroinit::Bool = true)
    @assert all(==(num_dofs(field, grid)), size(A))
    zeroinit && fillzero!(A)
    for eltindex in 1:num_elements(grid)
        Ke = f(eltindex, get_element(grid, eltindex))
        dofs = elementdofs(field, grid, eltindex)
        add!(A, dofs, Ke)
    end
    A
end

## integrate!
for (ArrayType, create_array) in ((:AbstractMatrix, :create_matrix), (:AbstractVector, :create_vector))
    @eval function integrate!(f, A::$ArrayType, field::Field, grid::Grid; zeroinit::Bool = true)
        Ke = $create_array(eltype(A), field, get_element(grid))
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
    F = integrate_function(f, get_element(grid))
    F(f, field, grid)
end

#########################
# generate_elementstate #
#########################

function generate_elementstate(::Type{ElementState}, grid::Grid) where {ElementState}
    elementstate = StructArray{ElementState}(undef, num_quadpoints(get_element(grid)), num_elements(grid))
    fillzero!(elementstate)
    if :x in propertynames(elementstate)
        elementstate.x .= interpolate(grid, get_allnodes(grid))
    end
    elementstate
end
