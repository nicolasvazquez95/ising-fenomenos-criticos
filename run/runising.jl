# 1. Imports SIEMPRE a nivel global (top-level)
using Plots, Measures
using Random

# Cargamos ising.jl usando joinpath para evitar fallos según dónde abras la terminal
include(joinpath(@__DIR__, "..", "src", "ising.jl"))

# Observables a partir de la matriz de espines
function magnetization(s::AbstractMatrix)
    return sum(s) / length(s)
end

function energy(s::AbstractMatrix, J::Real, h::Real)
    m, n = size(s)
    E = 0.0
    @inbounds for j in 1:n
        for i in 1:m
            # Vecinos con condiciones periódicas
            NN = s[ifelse(i == 1, m, i - 1), j]
            SS = s[ifelse(i == m, 1, i + 1), j]
            WW = s[i, ifelse(j == 1, n, j - 1)]
            EE = s[i, ifelse(j == n, 1, j + 1)]
            
            # Dividido por 2 para no duplicar enlaces (edges)
            E -= 0.5 * J * s[i, j] * (NN + SS + WW + EE)
            E -= h * s[i, j]
        end
    end
    return E
end

function run_ising_simulation(L, T, J=1.0, h=0.0, n_snapshots=1000, master_seed=42; mcs_between=5)
    rng = Xoshiro(master_seed)
    
    # Inicialización aleatoria (Hot Start)
    s = rand(rng, Int8[-1, 1], L, L)
    s0 = copy(s)

    magnetizations = zeros(Float64, n_snapshots)
    energies = zeros(Float64, n_snapshots)

    for snap in 1:n_snapshots
        # mcs_between barridos completos entre cada medición
        # Pasamos el argumento de la semilla como posicional o usamos el mismo flujo
        ising2d!(s, J, h, T, mcs_between, rand(rng, UInt64))

        magnetizations[snap] = magnetization(s)
        energies[snap] = energy(s, J, h)
    end

    return Dict(
        :spins0 => s0,
        :spinsf => s,
        :mag    => magnetizations,
        :ene    => energies
    )
end

# Test y verificación
function test_run_ising_simulation()
    L = 16
    T = 2.269
    J = 1.0
    h = 0.0
    n_snapshots = 500
    master_seed = 42

    result = run_ising_simulation(L, T, J, h, n_snapshots, master_seed; mcs_between=1)

    @assert size(result[:spins0]) == (L, L)
    @assert size(result[:spinsf]) == (L, L)
    @assert length(result[:mag]) == n_snapshots
    @assert length(result[:ene]) == n_snapshots

    println("✓ Test passed: run_ising_simulation ejecutado con éxito.")

    # Graficar
    p1 = plot(result[:mag], title="Magnetización vs Snapshot", xlabel="Snapshot", ylabel="m", label="m(t)", lw=1.5)
    p2 = plot(result[:ene], title="Energía vs Snapshot", xlabel="Snapshot", ylabel="E", label="E(t)", lw=1.5, color=:red)
    
    fig = plot(p1, p2, layout=(2, 1), size=(700, 500), margin=5mm, dpi=150)
    # Save in the same directory as the script
    savefig(fig, joinpath(@__DIR__, "data/ising_snapshots_test.png"))
    println("✓ Gráfico combinado guardado como 'ising_snapshots_test.png'.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_run_ising_simulation()
end