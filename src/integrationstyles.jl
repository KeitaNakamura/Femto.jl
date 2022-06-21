###############
# TensorStyle #
###############

struct TensorStyle{vector_or_matrix} <: IntegrationStyle end

function IntegrationStyle(f, fieldtype::FieldType, element::Element)
    IntegrationStyle(f, typeof(element))
end

@pure function IntegrationStyle(f, ::Type{<: BodyElement})
    nargs = last(methods(f)).nargs - 1
    nargs == 2 && return TensorStyle{:vector}()
    nargs == 3 && return TensorStyle{:matrix}()
    error("wrong number of arguments in `integrate`, use `(index, u, v)` for matrix or `(index, v)` for
 vector")
end

@pure function IntegrationStyle(f, ::Type{<: FaceElement})
    nargs = last(methods(f)).nargs - 1
    nargs == 3 && return TensorStyle{:vector}()
    error("wrong number of arguments in `integrate`, use `(index, v, normal)`")
end

@generated function build_element_vector(f, qp, N::SVector{L}, args...) where {L}
    exps = [:(f(qp, N[$i], args...)) for i in 1:L]
    quote
        @_inline_meta
        @inbounds SVector{$L}($(exps...))
    end
end

@generated function build_element_matrix(f, qp, v::SVector{L1}, u::SVector{L2}) where {L1, L2}
    exps = [:(f(qp, u[$j], v[$i])) for i in 1:L1, j in 1:L2]
    quote
        @_inline_meta
        @inbounds SMatrix{$L1, $L2}($(exps...))
    end
end

@inline function build_element(f, ::TensorStyle{:vector}, fieldtype::FieldType, element::FaceElement, qp::Int)
    @_propagate_inbounds_meta
    v = function_values(fieldtype, element, qp)
    detJdΩ = element.detJdΩ[qp]
    normal = element.normal[qp]
    build_element_vector(f, qp, v, normal) * detJdΩ
end

@inline function build_element(f, ::TensorStyle{vector_or_matrix}, fieldtype::FieldType, element::Element, qp::Int) where {vector_or_matrix}
    @_propagate_inbounds_meta
    N = function_values(fieldtype, element, qp)
    dNdx = function_gradients(fieldtype, element, qp)
    detJdΩ = element.detJdΩ[qp]
    u = v = map(dual, N, dNdx)
    vector_or_matrix == :vector && return build_element_vector(f, qp, v)    * detJdΩ
    vector_or_matrix == :matrix && return build_element_matrix(f, qp, v, u) * detJdΩ
    error("unreachable")
end
