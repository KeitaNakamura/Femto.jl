using Femto

function HeatEquation(filename = joinpath(@__DIR__, "model2d.msh"))
    HeatEquation(readgmsh(filename), dirname(filename))
end

function HeatEquation(gridset::Dict, dir::String = @__DIR__)
    grid = gridset["main"]

    K = integrate((index,v,u) -> ∇(v) ⋅ ∇(u), Sf(), Sf(), grid)
    F = integrate((index,v) -> v, Sf(), grid)

    U = zeros(length(F))
    dirichlet = falses(length(U))
    dirichlet[nodedofs(Sf(), gridset["boundary"])] .= true

    solve!(U, K, F, dirichlet)

    openvtk(joinpath(dir, "HeatEquation"), grid) do vtk
        vtk["Temperature"] = U
    end

    U
end
