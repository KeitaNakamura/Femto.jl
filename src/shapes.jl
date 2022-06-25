abstract type Shape{dim} end

get_local_node_coordinates(s::Shape) = get_local_node_coordinates(Float64, s)
get_dimension(s::Shape{dim}) where {dim} = dim
quadpoints(s::Shape) = quadpoints(Float64, s)
quadweights(s::Shape) = quadweights(Float64, s)

# values_gradients
@inline function values_gradients(shape::Shape, X::Vec)
    grads, vals = gradient(X -> Tensor(values(shape, X)), X, :all)
    SVector(Tuple(vals)), _reinterpret_to_vec(grads)
end
@inline function _reinterpret_to_vec(A::Mat{N, dim}) where {N, dim}
    SVector(ntuple(i->A[i,:], Val(N)))
end

"""
    Line2()

# Geometry
```
      η
      ^
      |
      |
1-----+-----2 --> ξ
```
"""
struct Line2 <: Shape{1} end

@pure num_nodes(::Line2) = 2
@pure num_quadpoints(::Line2) = 1

function get_local_node_coordinates(::Type{T}, ::Line2) where {T}
    SVector{2, Vec{1, T}}(
        (-1.0,),
        ( 1.0,),
    )
end

function Base.values(::Line2, X::Vec{1, T}) where {T}
    ξ = X[1]
    SVector{2, T}(
        0.5 * (1-ξ),
        0.5 * (1+ξ),
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

"""
    Line3()

# Geometry
```
      η
      ^
      |
      |
1-----3-----2 --> ξ
```
"""
struct Line3 <: Shape{1} end

@pure num_nodes(::Line3) = 3
@pure num_quadpoints(::Line3) = 2

function get_local_node_coordinates(::Type{T}, ::Line3) where {T}
    SVector{3, Vec{1, T}}(
        (-1.0,),
        ( 1.0,),
        ( 0.0,),
    )
end

function Base.values(::Line3, X::Vec{1, T}) where {T}
    ξ = X[1]
    SVector{3, T}(
        -0.5 * ξ*(1-ξ),
         0.5 * ξ*(1+ξ),
        1 - ξ^2,
    )
end

function quadpoints(::Type{T}, ::Line3) where {T}
    ξ = √3 / 3
    NTuple{2, Vec{1, T}}((
        (-ξ,),
        ( ξ,),
    ))
end
function quadweights(::Type{T}, ::Line3) where {T}
    NTuple{2, T}((1, 1))
end

"""
    Quad4()

# Geometry
```
      η
      ^
      |
4-----------3
|     |     |
|     |     |
|     +---- | --> ξ
|           |
|           |
1-----------2
```
"""
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

function Base.values(::Quad4, X::Vec{2, T}) where {T}
    ξ, η = X
    SVector{4, T}(
        0.25 * (1-ξ) * (1-η),
        0.25 * (1+ξ) * (1-η),
        0.25 * (1+ξ) * (1+η),
        0.25 * (1-ξ) * (1+η),
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

"""
    Quad9()

# Geometry
```
      η
      ^
      |
4-----7-----3
|     |     |
|     |     |
8     9---- 6 --> ξ
|           |
|           |
1-----5-----2
```
"""
struct Quad9 <: Shape{2} end

@pure num_nodes(::Quad9) = 9
@pure num_quadpoints(::Quad9) = 9

function get_local_node_coordinates(::Type{T}, ::Quad9) where {T}
    SVector{9, Vec{2, T}}(
        (-1.0, -1.0),
        ( 1.0, -1.0),
        ( 1.0,  1.0),
        (-1.0,  1.0),
        ( 0.0, -1.0),
        ( 1.0,  0.0),
        ( 0.0,  1.0),
        (-1.0,  0.0),
        ( 0.0,  0.0),
    )
end

function Base.values(::Quad9, X::Vec{2, T}) where {T}
    ξ, η = X
    SVector{9, T}(
         0.25 * ξ * η * (1-ξ) * (1-η),
        -0.25 * ξ * η * (1+ξ) * (1-η),
         0.25 * ξ * η * (1+ξ) * (1+η),
        -0.25 * ξ * η * (1-ξ) * (1+η),
        -0.5 * η * (1-ξ^2) * (1-η),
         0.5 * ξ * (1+ξ) * (1-η^2),
         0.5 * η * (1-ξ^2) * (1+η),
        -0.5 * ξ * (1-ξ) * (1-η^2),
         (1-ξ^2) * (1-η^2),
    )
end

function quadpoints(::Type{T}, ::Quad9) where {T}
    ξ = η = √(3/5)
    NTuple{9, Vec{2, T}}((
        (-ξ, -η),
        ( 0, -η),
        ( ξ, -η),
        (-ξ,  0),
        ( 0,  0),
        ( ξ,  0),
        (-ξ,  η),
        ( 0,  η),
        ( ξ,  η),
    ))
end
function quadweights(::Type{T}, ::Quad9) where {T}
    NTuple{9, T}((
        5/9 * 5/9,
        8/9 * 5/9,
        5/9 * 5/9,
        5/9 * 8/9,
        8/9 * 8/9,
        5/9 * 8/9,
        5/9 * 5/9,
        8/9 * 5/9,
        5/9 * 5/9,
    ))
end

raw"""
    Hex8()

# Geometry
```
       η
4----------3
|\     ^   |\
| \    |   | \
|  \   |   |  \
|   8------+---7
|   |  +-- |-- | -> ξ
1---+---\--2   |
 \  |    \  \  |
  \ |     \  \ |
   \|      ζ  \|
    5----------6
```
"""
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

function Base.values(::Hex8, X::Vec{3, T}) where {T}
    ξ, η, ζ = X
    SVector{8, T}(
        0.125 * (1-ξ) * (1-η) * (1-ζ),
        0.125 * (1+ξ) * (1-η) * (1-ζ),
        0.125 * (1+ξ) * (1+η) * (1-ζ),
        0.125 * (1-ξ) * (1+η) * (1-ζ),
        0.125 * (1-ξ) * (1-η) * (1+ζ),
        0.125 * (1+ξ) * (1-η) * (1+ζ),
        0.125 * (1+ξ) * (1+η) * (1+ζ),
        0.125 * (1-ξ) * (1+η) * (1+ζ),
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

@doc raw"""
    Hex27()

# Geometry
```
       η
       ^
4-----7+---3
|\     |   |\
|22    25  | 21
8  \  9|   6  \
|  13----16+---12
|26 |  27--|-24| -> ξ
1---+-5-\--2   |
 \ 17    18 \  15
 19 |  23 \  20|
   \|      ζ  \|
   10----14----11
```
"""
struct Hex27 <: Shape{3} end

@pure num_nodes(::Hex27) = 27
@pure num_quadpoints(::Hex27) = 27

function get_local_node_coordinates(::Type{T}, ::Hex27) where {T}
    SVector{27, Vec{3, T}}(
        (-1.0, -1.0, -1.0),
        ( 1.0, -1.0, -1.0),
        ( 1.0,  1.0, -1.0),
        (-1.0,  1.0, -1.0),
        ( 0.0, -1.0, -1.0),
        ( 1.0,  0.0, -1.0),
        ( 0.0,  1.0, -1.0),
        (-1.0,  0.0, -1.0),
        ( 0.0,  0.0, -1.0),
        (-1.0, -1.0,  1.0),
        ( 1.0, -1.0,  1.0),
        ( 1.0,  1.0,  1.0),
        (-1.0,  1.0,  1.0),
        ( 0.0, -1.0,  1.0),
        ( 1.0,  0.0,  1.0),
        ( 0.0,  1.0,  1.0),
        (-1.0,  0.0,  1.0),
        ( 0.0,  0.0,  1.0),
        (-1.0, -1.0,  0.0),
        ( 1.0, -1.0,  0.0),
        ( 1.0,  1.0,  0.0),
        (-1.0,  1.0,  0.0),
        ( 0.0, -1.0,  0.0),
        ( 1.0,  0.0,  0.0),
        ( 0.0,  1.0,  0.0),
        (-1.0,  0.0,  0.0),
        ( 0.0,  0.0,  0.0),
    )
end

function Base.values(::Hex27, X::Vec{3, T}) where {T}
    ξ, η, ζ = X
    SVector{27, T}(
        -0.125 * ξ*η*ζ * (1-ξ) * (1-η) * (1-ζ),
         0.125 * ξ*η*ζ * (1+ξ) * (1-η) * (1-ζ),
        -0.125 * ξ*η*ζ * (1+ξ) * (1+η) * (1-ζ),
         0.125 * ξ*η*ζ * (1-ξ) * (1+η) * (1-ζ),
         0.25 * η*ζ * (1-ξ^2) * (1-η) * (1-ζ),
        -0.25 * ξ*ζ * (1+ξ) * (1-η^2) * (1-ζ),
        -0.25 * η*ζ * (1-ξ^2) * (1+η) * (1-ζ),
         0.25 * ξ*ζ * (1-ξ) * (1-η^2) * (1-ζ),
        -0.5 * ζ * (1-ξ^2) * (1-η^2) * (1-ζ),
         0.125 * ξ*η*ζ * (1-ξ) * (1-η) * (1+ζ),
        -0.125 * ξ*η*ζ * (1+ξ) * (1-η) * (1+ζ),
         0.125 * ξ*η*ζ * (1+ξ) * (1+η) * (1+ζ),
        -0.125 * ξ*η*ζ * (1-ξ) * (1+η) * (1+ζ),
        -0.25 * η*ζ * (1-ξ^2) * (1-η) * (1+ζ),
         0.25 * ξ*ζ * (1+ξ) * (1-η^2) * (1+ζ),
         0.25 * η*ζ * (1-ξ^2) * (1+η) * (1+ζ),
        -0.25 * ξ*ζ * (1-ξ) * (1-η^2) * (1+ζ),
         0.5 * ζ * (1-ξ^2) * (1-η^2) * (1+ζ),
         0.25 * ξ*η * (1-ξ) * (1-η) * (1-ζ^2),
        -0.25 * ξ*η * (1+ξ) * (1-η) * (1-ζ^2),
         0.25 * ξ*η * (1+ξ) * (1+η) * (1-ζ^2),
        -0.25 * ξ*η * (1-ξ) * (1+η) * (1-ζ^2),
        -0.5 * η * (1-ξ^2) * (1-η) * (1-ζ^2),
         0.5 * ξ * (1+ξ) * (1-η^2) * (1-ζ^2),
         0.5 * η * (1-ξ^2) * (1+η) * (1-ζ^2),
        -0.5 * ξ * (1-ξ) * (1-η^2) * (1-ζ^2),
         (1-ξ^2) * (1-η^2) * (1-ζ^2),
    )
end

function quadpoints(::Type{T}, ::Hex27) where {T}
    ξ = η = ζ = √(3/5)
    NTuple{27, Vec{3, T}}((
        (-ξ, -η, -ζ),
        ( 0, -η, -ζ),
        ( ξ, -η, -ζ),
        (-ξ,  0, -ζ),
        ( 0,  0, -ζ),
        ( ξ,  0, -ζ),
        (-ξ,  η, -ζ),
        ( 0,  η, -ζ),
        ( ξ,  η, -ζ),
        (-ξ, -η,  0),
        ( 0, -η,  0),
        ( ξ, -η,  0),
        (-ξ,  0,  0),
        ( 0,  0,  0),
        ( ξ,  0,  0),
        (-ξ,  η,  0),
        ( 0,  η,  0),
        ( ξ,  η,  0),
        (-ξ, -η,  η),
        ( 0, -η,  η),
        ( ξ, -η,  η),
        (-ξ,  0,  η),
        ( 0,  0,  η),
        ( ξ,  0,  η),
        (-ξ,  η,  η),
        ( 0,  η,  η),
        ( ξ,  η,  η),
    ))
end
function quadweights(::Type{T}, ::Hex27) where {T}
    NTuple{27, T}((
        5/9 * 5/9 * 5/9,
        8/9 * 5/9 * 5/9,
        5/9 * 5/9 * 5/9,
        5/9 * 8/9 * 5/9,
        8/9 * 8/9 * 5/9,
        5/9 * 8/9 * 5/9,
        5/9 * 5/9 * 5/9,
        8/9 * 5/9 * 5/9,
        5/9 * 5/9 * 5/9,
        5/9 * 5/9 * 8/9,
        8/9 * 5/9 * 8/9,
        5/9 * 5/9 * 8/9,
        5/9 * 8/9 * 8/9,
        8/9 * 8/9 * 8/9,
        5/9 * 8/9 * 8/9,
        5/9 * 5/9 * 8/9,
        8/9 * 5/9 * 8/9,
        5/9 * 5/9 * 8/9,
        5/9 * 5/9 * 5/9,
        8/9 * 5/9 * 5/9,
        5/9 * 5/9 * 5/9,
        5/9 * 8/9 * 5/9,
        8/9 * 8/9 * 5/9,
        5/9 * 8/9 * 5/9,
        5/9 * 5/9 * 5/9,
        8/9 * 5/9 * 5/9,
        5/9 * 5/9 * 5/9,
    ))
end

@doc raw"""
    Tri3()

# Geometry
```
η
^
|
2
|`\
|  `\
|    `\
|      `\
|        `\
3----------1 --> ξ
```
"""
struct Tri3 <: Shape{2} end

@pure num_nodes(::Tri3) = 3
@pure num_quadpoints(::Tri3) = 1

function get_local_node_coordinates(::Type{T}, ::Tri3) where {T}
    SVector{3, Vec{2, T}}(
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 0.0),
    )
end

function Base.values(::Tri3, X::Vec{2, T}) where {T}
    ξ, η = X
    ζ = 1 - ξ - η
    SVector{3, T}(ξ, η, ζ)
end

function quadpoints(::Type{T}, ::Tri3) where {T}
    NTuple{1, Vec{2, T}}((
        (1/3, 1/3),
    ))
end
function quadweights(::Type{T}, ::Tri3) where {T}
    NTuple{1, T}((0.5,))
end

@doc raw"""
    Tri6()

# Geometry
```
η
^
|
2
|`\
|  `\
5    `4
|      `\
|        `\
3-----6----1 --> ξ
```
"""
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

function Base.values(::Tri6, X::Vec{2, T}) where {T}
    ξ, η = X
    ζ = 1 - ξ - η
    SVector{6, T}(
        ξ * (2ξ-1),
        η * (2η-1),
        ζ * (2ζ-1),
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
