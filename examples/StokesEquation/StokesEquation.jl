using Femto

function StokesEquation(gridset::Dict; dir::String = @__DIR__)
    field = mixed(VectorField(2), ScalarField(1))

    grid = gridset["main"]
    ndofs = num_dofs(field, grid)

    K = integrate((index, (v,q), (u,p)) -> ∇(v) ⊡ ∇(u) - (∇⋅v)*p + q*(∇⋅u), field, grid)
    F = zeros(ndofs)
    U = zeros(ndofs)
    dirichlet = falses(ndofs)

    for dofs in get_nodedofs(field, gridset["top"])[1]
        U[dofs] = Vec(1,0)
        dirichlet[dofs] .= true
    end
    for name in ("left", "bottom", "right")
        for dofs in get_nodedofs(field, gridset[name])[1]
            U[dofs] = Vec(0,0)
            dirichlet[dofs] .= true
        end
    end

    solve!(U, K, F, dirichlet)

    openvtk(joinpath(dir, "StokesEquation"), decrease_order(grid)) do vtk
        U_u, U_p = gridvalues(U, field, grid)
        vtk["Velocity"] = view(U_u, 1:length(U_p))
        vtk["Pressure"] = U_p
    end

    U
end

StokesEquation(; dir::String = @__DIR__) = StokesEquation(readgmsh(joinpath(@__DIR__, "model.msh")); dir)
