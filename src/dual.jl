##################
# ScalarGradient #
##################

struct ScalarGradient{T, G} <: Real
    scalar::T
    gradient::G
end

Base.promote_rule(::Type{<: ScalarGradient{T}}, ::Type{U}) where {T <: Real, U <: Real} = promote_type(T, U)
Base.convert(::Type{T}, x::ScalarGradient) where {T <: Real} = convert(T, x.scalar)
Base.convert(::Type{ScalarGradient{T, G}}, x::ScalarGradient{T, G}) where {T <: Real, G} = x
Base.float(x::ScalarGradient) = float(x.scalar)
Base.zero(::Type{ScalarGradient{T, G}}) where {T, G} = ScalarGradient(zero(T), zero(G))
Base.one(::Type{ScalarGradient{T, G}}) where {T, G} = one(T)

for op in (:+, :-, :*, :/)
    @eval Base.$op(a::ScalarGradient, b::ScalarGradient) = $op(a.scalar, b.scalar)
end

∇(x::ScalarGradient) = x.gradient

function Base.show(io::IO, x::ScalarGradient)
    print(io, x.scalar, " (∇x = ", x.gradient, ")")
end

##################
# TensorGradient #
##################

struct TensorGradient{S, T, N, TT <: Tensor{S, T, N}, G} <: AbstractTensor{S, T, N}
    tensor::TT
    gradient::G
end
Base.Tuple(x::TensorGradient) = Tuple(x.tensor)
@inline function Base.getindex(x::TensorGradient, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds x.tensor[i]
end
Base.zero(::Type{TensorGradient{S, T, N, TT, G}}) where {S, T, N, TT, G} = TensorGradient(zero(TT), zero(G))

∇(x::TensorGradient) = x.gradient
LinearAlgebra.dot(::typeof(∇), x::TensorGradient{Tuple{dim}}) where {dim} = tr(∇(x))

function Base.show(io::IO, ::MIME"text/plain", x::TensorGradient)
    print(io, x.tensor, " (∇x = ", x.gradient, ")")
end

########
# dual #
########

dual_gradient(u::Real, dudx::Vec) = ScalarGradient(u, dudx)
dual_gradient(u::Tensor, dudx::Tensor) = TensorGradient(u, dudx)
