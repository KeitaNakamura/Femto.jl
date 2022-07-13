using Femto

function HeatEquation(filename = joinpath(@__DIR__, "model.msh"))
    HeatEquation(readgmsh(filename), dirname(filename))
end
function HeatEquation(gridset::Dict, dir::String = @__DIR__)
    grid = gridset["main"]
    fieldtype = ScalarField()

    K = integrate((index,u,v) -> ∇(v) ⋅ ∇(u), fieldtype, grid)
    F = integrate((index,v) -> v, fieldtype, grid)

    U = zeros(length(F))
    dirichlet = falses(length(U))
    dirichlet[eachnode(fieldtype, gridset["boundary"])] .= true

    solve!(U, K, F, dirichlet)

    openvtk(joinpath(dir, "HeatEquation"), grid) do vtk
        vtk["Temperature"] = U
    end

    U
end
