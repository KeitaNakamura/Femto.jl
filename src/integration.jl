function integrate_matrix(f, args...; kwargs...)
    T = infer_integrate_matrix_eltype(f, args...)
    A = create_matrix(T, args...)
    integrate!(f, A, args...; kwargs...)
end

function integrate_vector(f, args...; kwargs...)
    T = infer_integrate_vector_eltype(f, args...)
    A = create_vector(T, args...)
    integrate!(f, A, args...; kwargs...)
end
