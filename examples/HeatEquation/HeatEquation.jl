using Femto

function HeatEquation(filename = joinpath(@__DIR__, "model.msh"))
    gridset = readgmsh(filename)

    grid = gridset["main"]
    fieldtype = ScalarField()

    K = integrate((index,u,v) -> ∇(v) ⋅ ∇(u), fieldtype, grid)
    F = integrate((index,v) -> v, fieldtype, grid)

    U = zeros(length(F))
    dirichlet = falses(length(U))
    dirichlet[eachnode(fieldtype, gridset["boundary"])] .= true

    solve!(U, K, F, dirichlet)

    openvtk(joinpath(@__DIR__, "HeatEquation"), grid) do vtk
        vtk["Temperature"] = U
    end

    U
end
