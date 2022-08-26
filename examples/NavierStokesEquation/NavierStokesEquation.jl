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
    A = SparseMatrixCOO(ndofs, ndofs)
    U = zeros(ndofs)
    Uₙ = zeros(ndofs)
    dU = zeros(ndofs)
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

        nlsolve!(U, dirichlet) do R, J, U
            R .= (M * (U - Uₙ)) / dt + K * U
            if autodiff
                integrate!(R, A, field, grid, U; zeroinit=(false,true)) do i, (v,q), (u,p)
                    v ⋅ (∇(u) ⋅ u)
                end
            else
                Ũ = collect(interpolate(field, grid, U))
                integrate!(R, field, grid; zeroinit=false) do i, (v,q)
                    ũ, p̃ = Ũ[i]
                    v ⋅ (∇(ũ) ⋅ ũ)
                end
                integrate!(A, field, grid) do i, (v,q), (u,p)
                    ũ, p̃ = Ũ[i]
                    v ⋅ (∇(ũ)⋅u + ∇(u)⋅ũ)
                end
            end
            J .= sparse(A) + M/dt + K
        end
        Uₙ .= U

        if step % 10 == 0
            openpvd(pvdfile; append = true) do pvd
                grid_lower = decrease_order(grid)
                openvtk(joinpath(outdir, "NavierStokesEquation_$step"), grid_lower) do vtk
                    U_u, U_p = gridvalues(U, field, grid)
                    # compute vorticity by L2 projection
                    Ũ = interpolate(field, grid, U)
                    ωV = integrate(ScalarField(), grid_lower) do i, ϕ
                        ũ, p̃ = Ũ[i]
                        (∇(ũ)[1,2] - ∇(ũ)[2,1]) * ϕ
                    end
                    V = integrate((i,ϕ)->ϕ, ScalarField(), grid_lower)
                    # save to vtk
                    vtk["Velocity"] = view(U_u, 1:length(U_p))
                    vtk["Pressure"] = U_p
                    vtk["Vorticity"] = ωV ./ V
                    pvd[t] = vtk
                end
            end
        end
    end

    U
end
