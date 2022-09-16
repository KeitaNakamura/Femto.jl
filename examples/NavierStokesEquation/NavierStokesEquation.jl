using Femto

function NavierStokesEquation(
        gridset::Dict = readgmsh(joinpath(@__DIR__, "model.msh"));
        dir::String = @__DIR__,
        t_stop::Real = 10.0,
        autodiff::Bool = true,
    )
    field = mixed(VectorField(2), ScalarField(1))

    grid = gridset["main"]
    ndofs = num_dofs(field, grid)
    v_max = 1.5
    ν = 1e-3 # kinematic viscosity

    K = sparse(integrate((index, (v,q), (u,p)) -> ν * ∇(v) ⊡ ∇(u) - (∇⋅v)*p + q*(∇⋅u), field, grid))
    M = sparse(integrate((index, (v,q), (u,p)) -> v ⋅ u, field, grid))
    U = zeros(ndofs)
    Uₙ = zeros(ndofs)
    dirichlet = falses(ndofs)

    outdir = mkpath(joinpath(dir, "Output"))
    pvdfile = joinpath(outdir, "NavierStokesEquation")
    closepvd(openpvd(pvdfile))

    timespan = 0.0:0.01:t_stop
    dt = step(timespan)

    for (step, t) in enumerate(timespan)
        for dofs in get_nodedofs(field, gridset["left"])[1]
            nodes = get_allnodes_flatten(grid)
            x, y = nodes[dofs]
            U[dofs] = Vec(4v_max*y*(0.41-y)/0.41^2, 0) # parabolic inflow
            dirichlet[dofs] .= true
        end
        for dofs in get_nodedofs(field, gridset["no-slip boundary"])[1]
            U[dofs] = Vec(0,0)
            dirichlet[dofs] .= true
        end

        nlsolve!(U, dirichlet; sparsity_pattern=K) do R, J, U
            if autodiff
                integrate!(R, J, field, grid, U) do i, (v,q), (u,p)
                    v ⋅ (∇(u) ⋅ u)
                end
            else
                Ũ = collect(interpolate(field, grid, U))
                integrate!(R, field, grid) do i, (v,q)
                    ũ, p̃ = Ũ[i]
                    v ⋅ (∇(ũ) ⋅ ũ)
                end
                integrate!(J, field, grid) do i, (v,q), (u,p)
                    ũ, p̃ = Ũ[i]
                    v ⋅ (∇(ũ)⋅u + ∇(u)⋅ũ)
                end
            end
            R .= R + (M * (U - Uₙ)) / dt + K * U
            J .= J + M/dt + K
        end
        Uₙ .= U

        if step % 10 == 0
            openpvd(pvdfile; append = true) do pvd
                openvtk(joinpath(outdir, "NavierStokesEquation_$step"), grid) do vtk
                    U_u, U_p = gridvalues(U, field, grid)
                    # compute pressure and vorticity by L2 projection
                    Ũ = collect(interpolate(field, grid, U))
                    V = integrate((i,ϕ)->1.0, ScalarField(), grid)
                    pV = integrate(ScalarField(), grid) do i, ϕ
                        ũ, p̃ = Ũ[i]
                        p̃
                    end
                    ωV = integrate(ScalarField(), grid) do i, ϕ
                        ũ, p̃ = Ũ[i]
                        ∇(ũ)[1,2] - ∇(ũ)[2,1]
                    end
                    # save to vtk
                    vtk["Velocity"] = U_u
                    vtk["Pressure"] = pV ./ V
                    vtk["Vorticity"] = ωV ./ V
                    pvd[t] = vtk
                end
            end
        end
    end

    U
end
