using Femto

function HeatEquation(filename = joinpath(@__DIR__, "model2d_1.msh"))
    HeatEquation(readgmsh(filename); dir = dirname(filename))
end

function HeatEquation(field::Field, gridset::Dict; dir::String = @__DIR__)
    grid = gridset["main"]

    K = integrate((index,v,u) -> ∇(v) ⋅ ∇(u), field, grid)
    F = integrate((index,v) -> v, field, grid)

    U = zeros(length(F))
    dirichlet = falses(length(U))
    dirichlet[get_nodedofs(field, gridset["boundary"])] .= true

    solve!(U, K, F, dirichlet)

    openvtk(joinpath(dir, "HeatEquation"), field, grid) do vtk
        vtk["Temperature"] = U
    end

    U
end

HeatEquation(gridset::Dict; dir::String = @__DIR__) = HeatEquation(ScalarField(), gridset; dir)
