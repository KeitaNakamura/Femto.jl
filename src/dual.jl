###################################
# ScalarGradient/ScalarProjection #
###################################

struct ScalarGradient{T <: Real, D} <: Real
    scalar::T
    gradient::D
end
struct ScalarProjection{T <: Real, D} <: Real
    scalar::T
    projection::D
end

for DT in (:ScalarGradient, :ScalarProjection)
    @eval begin
        Base.promote_rule(::Type{<: $DT{T}}, ::Type{U}) where {T, U <: Real} = promote_type(T, U)
        Base.convert(::Type{T}, x::$DT) where {T <: Real} = convert(T, x.scalar)
        Base.convert(::Type{$DT{T, D}}, x::$DT{T, D}) where {T, D} = x
        Base.zero(::Type{$DT{T, D}}) where {T, D} = $DT(zero(T), zero(D))
        Base.AbstractFloat(x::$DT) = AbstractFloat(x.scalar)
    end
    for op in (:+, :-, :*, :/)
        @eval Base.$op(a::$DT, b::$DT) = $op(a.scalar, b.scalar)
    end
end

∇(x::ScalarGradient) = x.gradient
Π(x::ScalarProjection) = x.projection

function Base.show(io::IO, x::ScalarGradient)
    print(io, x.scalar, " (∇(x) = ", x.gradient, ")")
end
function Base.show(io::IO, x::ScalarProjection)
    print(io, x.scalar, " (Π(x) = ", x.projection, ")")
end

###################################
# TensorGradient/TensorProjection #
###################################

struct TensorGradient{S, T, N, TT <: Tensor{S, T, N}, D} <: AbstractTensor{S, T, N}
    tensor::TT
    gradient::D
end
struct TensorProjection{S, T, N, TT <: Tensor{S, T, N}, D} <: AbstractTensor{S, T, N}
    tensor::TT
    projection::D
end

for DT in (:TensorGradient, :TensorProjection)
    @eval begin
        Base.Tuple(x::$DT) = Tuple(x.tensor)
        @inline function Base.getindex(x::$DT, i::Int)
            @boundscheck checkbounds(x, i)
            @inbounds x.tensor[i]
        end
        Base.zero(::Type{$DT{S, T, N, TT, D}}) where {S, T, N, TT, D} = $DT(zero(TT), zero(D))
    end
end

∇(x::TensorGradient) = x.gradient
Π(x::TensorProjection) = x.projection
LinearAlgebra.dot(::typeof(∇), x::TensorGradient{Tuple{dim}}) where {dim} = tr(∇(x))

function Base.show(io::IO, ::MIME"text/plain", x::TensorGradient)
    print(io, x.tensor, " (∇(x) = ", x.gradient, ")")
end
function Base.show(io::IO, ::MIME"text/plain", x::TensorProjection)
    print(io, x.tensor, " (Π(x) = ", x.projection, ")")
end

########
# dual #
########

dual_gradient(u::Real, dudx::Vec) = ScalarGradient(u, dudx)
dual_gradient(u::Tensor, dudx::Tensor) = TensorGradient(u, dudx)

dual_projection(u::Real, Πu::Real) = ScalarProjection(u, Πu)
dual_projection(u::Tensor, Πu::Tensor) = TensorProjection(u, Πu)
