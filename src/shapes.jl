abstract type Shape{dim} end

get_local_node_coordinates(s::Shape) = get_local_node_coordinates(Float64, s)
get_dimension(s::Shape{dim}) where {dim} = dim
quadpoints(s::Shape) = quadpoints(Float64, s)
quadweights(s::Shape) = quadweights(Float64, s)

# values_gradients
@inline function values_gradients(shape::Shape, X::Vec{dim}) where {dim}
    grads, vals = gradient(X -> Tensor(values(shape, X)), X, :all)
    SVector(Tuple(vals)), _reinterpret_to_vec(grads')
end
@generated function _reinterpret_to_vec(A::Mat{dim, N, T}) where {dim, N, T}
    exps = map(1:N) do j
        vals = [:(A[$i,$j]) for i in 1:dim]
        :(Vec($(vals...)))
    end
    quote
        @_inline_meta
        @inbounds SVector($(exps...))
    end
end

#########
# Line2 #
#########

struct Line2 <: Shape{1} end

@pure num_nodes(::Line2) = 2
@pure num_quadpoints(::Line2) = 1

function get_local_node_coordinates(::Type{T}, ::Line2) where {T}
    SVector{2, Vec{1, T}}(
        (-1.0,),
        ( 1.0,),
    )
end

function Base.values(::Line2, X::Vec{1})
    ξ = X[1]
    SVector{2}(
        (1 - ξ) / 2,
        (1 + ξ) / 2,
    )
end

function quadpoints(::Type{T}, ::Line2) where {T}
    NTuple{1, Vec{1, T}}((
        (0,),
    ))
end
function quadweights(::Type{T}, ::Line2) where {T}
    NTuple{1, T}((1,))
end

#########
# Quad4 #
#########

struct Quad4 <: Shape{2} end

@pure num_nodes(::Quad4) = 4
@pure num_quadpoints(::Quad4) = 4

function get_local_node_coordinates(::Type{T}, ::Quad4) where {T}
    SVector{4, Vec{2, T}}(
        (-1.0, -1.0),
        ( 1.0, -1.0),
        ( 1.0,  1.0),
        (-1.0,  1.0),
    )
end

function Base.values(::Quad4, X::Vec{2})
    ξ, η = X
    SVector{4}(
        (1 - ξ) * (1 - η) / 4,
        (1 + ξ) * (1 - η) / 4,
        (1 + ξ) * (1 + η) / 4,
        (1 - ξ) * (1 + η) / 4,
    )
end

function quadpoints(::Type{T}, ::Quad4) where {T}
    ξ = η = √3 / 3
    NTuple{4, Vec{2, T}}((
        (-ξ, -η),
        ( ξ, -η),
        ( ξ,  η),
        (-ξ,  η),
    ))
end
function quadweights(::Type{T}, ::Quad4) where {T}
    NTuple{4, T}((1, 1, 1, 1))
end

########
# Hex8 #
########

struct Hex8 <: Shape{3} end

@pure num_nodes(::Hex8) = 8
@pure num_quadpoints(::Hex8) = 8

function get_local_node_coordinates(::Type{T}, ::Hex8) where {T}
    SVector{8, Vec{3, T}}(
        (-1.0, -1.0, -1.0),
        ( 1.0, -1.0, -1.0),
        ( 1.0,  1.0, -1.0),
        (-1.0,  1.0, -1.0),
        (-1.0, -1.0,  1.0),
        ( 1.0, -1.0,  1.0),
        ( 1.0,  1.0,  1.0),
        (-1.0,  1.0,  1.0),
    )
end

function Base.values(::Hex8, X::Vec{3})
    ξ, η, ζ = X
    SVector{8}(
        (1 - ξ) * (1 - η) * (1 - ζ) / 8,
        (1 + ξ) * (1 - η) * (1 - ζ) / 8,
        (1 + ξ) * (1 + η) * (1 - ζ) / 8,
        (1 - ξ) * (1 + η) * (1 - ζ) / 8,
        (1 - ξ) * (1 - η) * (1 + ζ) / 8,
        (1 + ξ) * (1 - η) * (1 + ζ) / 8,
        (1 + ξ) * (1 + η) * (1 + ζ) / 8,
        (1 - ξ) * (1 + η) * (1 + ζ) / 8,
    )
end

function quadpoints(::Type{T}, ::Hex8) where {T}
    ξ = η = ζ = √3 / 3
    NTuple{8, Vec{3, T}}((
        (-ξ, -η, -ζ),
        ( ξ, -η, -ζ),
        ( ξ,  η, -ζ),
        (-ξ,  η, -ζ),
        (-ξ, -η,  ζ),
        ( ξ, -η,  ζ),
        ( ξ,  η,  ζ),
        (-ξ,  η,  ζ),
    ))
end
function quadweights(::Type{T}, ::Hex8) where {T}
    NTuple{8, T}((1, 1, 1, 1, 1, 1, 1, 1))
end

########
# Tri6 #
########

struct Tri6 <: Shape{2} end

@pure num_nodes(::Tri6) = 6
@pure num_quadpoints(::Tri6) = 3

function get_local_node_coordinates(::Type{T}, ::Tri6) where {T}
    SVector{6, Vec{2, T}}(
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (0.5, 0.5),
        (0.0, 0.5),
        (0.5, 0.0),
    )
end

function Base.values(::Tri6, X::Vec{2})
    ξ, η = X
    ζ = 1 - ξ - η
    SVector{6}(
        ξ * (2ξ - 1),
        η * (2η - 1),
        ζ * (2ζ - 1),
        4ξ * η,
        4η * ζ,
        4ζ * ξ,
    )
end

function quadpoints(::Type{T}, ::Tri6) where {T}
    NTuple{3, Vec{2, T}}((
        (1/6, 2/3),
        (1/6, 1/6),
        (2/3, 1/6),
    ))
end
function quadweights(::Type{T}, ::Tri6) where {T}
    NTuple{3, T}((1/3, 1/3, 1/3))
end
