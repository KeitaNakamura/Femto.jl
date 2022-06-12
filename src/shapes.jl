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
