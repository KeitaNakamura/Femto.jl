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
