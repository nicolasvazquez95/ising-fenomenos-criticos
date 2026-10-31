# run/hysteresis_loop.jl
#
# Ciclos de histeresis m(B) en Ising 2D: el campo se barre linealmente
# +B_max -> -B_max -> +B_max con un numero fijo de MCS por paso (fuera del
# equilibrio). Por debajo de Tc el sistema queda atrapado en la fase
# metaestable hasta que nuclea la fase estable, y el ciclo se abre.
#
# Se descarta el primer barrido descendente (desde la red saturada) y se
# promedia m(B) sobre n_cycles ciclos completos. Se guardan tambien
# instantaneas de la red cerca de m = 0 en la rama descendente.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Printf
using Dates
using Plots, Measures
using DataFrames
using CSV
using ArgParse
using Base.Threads
using Statistics

include(joinpath(@__DIR__, "runising.jl"))

# -------------------------------------------------------------------------
# Argumentos de línea de comandos
# -------------------------------------------------------------------------
function parse_commandline()
    s = ArgParseSettings(description = "Ciclos de histeresis m(B) en Ising 2D")

    @add_arg_table! s begin
        "--L"
            help = "Tamano lateral de la red (LxL)"
            arg_type = Int
            default = 64
        "--T_values"
            help = "Temperaturas, separadas por comas"
            arg_type = String
            default = "1.5,1.8,2.1,2.27,2.6"
        "--B_max"
            help = "Amplitud del barrido de campo B/J"
            arg_type = Float64
            default = 1.5
        "--n_steps"
            help = "Pasos de campo por rama (de +B_max a -B_max)"
            arg_type = Int
            default = 300
        "--mcs_per_step"
            help = "MCS por paso de campo (inversa de la velocidad de barrido)"
            arg_type = Int
            default = 10
        "--n_cycles"
            help = "Ciclos completos promediados (despues del barrido inicial descartado)"
            arg_type = Int
            default = 5
        "--out_dir"
            help = "Directorio de salida"
            arg_type = String
            default = joinpath(@__DIR__, "data")
        "--threads"
            help = "Numero maximo de hilos a utilizar"
            arg_type = Int
            default = nthreads()
    end

    return parse_args(s)
end

# -------------------------------------------------------------------------
# Ciclo de histeresis para una temperatura
# -------------------------------------------------------------------------
function hysteresis(L::Int, T::Float64, B_rama::Vector{Float64}, mcs::Int, n_cycles::Int, seed::Integer)
    J = 1.0
    rng = Xoshiro(seed)
    s = fill(Int8(1), L, L)  # Saturada en +B_max

    n = length(B_rama)
    m_baja = zeros(Float64, n)   # rama descendente (+B_max -> -B_max)
    m_sube = zeros(Float64, n)   # rama ascendente (-B_max -> +B_max)
    snapshot = copy(s)
    B_snapshot = NaN
    mejor = Inf

    # Barrido inicial descendente, descartado
    for B in B_rama
        ising2d!(s, J, B, T, mcs, rand(rng, UInt64))
    end
    # Barrido ascendente inicial para cerrar el primer ciclo
    for B in reverse(B_rama)
        ising2d!(s, J, B, T, mcs, rand(rng, UInt64))
    end

    for ciclo in 1:n_cycles
        for (k, B) in enumerate(B_rama)
            ising2d!(s, J, B, T, mcs, rand(rng, UInt64))
            m = magnetization(s)
            m_baja[k] += m / n_cycles
            # Instantanea del primer ciclo en el punto mas cercano a m = 0
            if ciclo == 1 && abs(m) < mejor
                mejor = abs(m)
                snapshot .= s
                B_snapshot = B
            end
        end
        for (k, B) in enumerate(reverse(B_rama))
            ising2d!(s, J, B, T, mcs, rand(rng, UInt64))
            m_sube[n + 1 - k] += magnetization(s) / n_cycles
        end
    end

    return m_baja, m_sube, snapshot, B_snapshot
end

"""Campo coercitivo: cruce de m = 0 por interpolacion lineal (NaN si no hay)."""
function coercive_field(B::Vector{Float64}, m::Vector{Float64})
    for k in 1:length(B) - 1
        if sign(m[k]) != sign(m[k + 1])
            return B[k] - m[k] * (B[k + 1] - B[k]) / (m[k + 1] - m[k])
        end
    end
    return NaN
end

# -------------------------------------------------------------------------
# Pipeline principal
# -------------------------------------------------------------------------
args = parse_commandline()
t_inicio = now()

L = args["L"]
temperaturas = parse.(Float64, strip.(split(args["T_values"], ",")))
B_max = args["B_max"]
n_steps = args["n_steps"]
mcs = args["mcs_per_step"]
n_cycles = args["n_cycles"]
dir_out = abspath(args["out_dir"])
active_threads = min(args["threads"], nthreads())
master_seed = 4242
mkpath(dir_out)

B_rama = collect(range(B_max, -B_max, length = n_steps))
nT = length(temperaturas)

println("="^72)
println(" CICLOS DE HISTERESIS - ISING 2D")
println("="^72)
println("Red L x L          : $L x $L")
println("Temperaturas       : $(join(temperaturas, ", "))")
@printf("Barrido de campo   : ±%.2f en %d pasos por rama, %d MCS por paso\n", B_max, n_steps, mcs)
@printf("Velocidad          : |dB/dt| = %.2e J/MCS\n", 2B_max / ((n_steps - 1) * mcs))
println("Ciclos promediados : $n_cycles (mas un ciclo inicial descartado)")
println("Hilos en uso       : $active_threads / $(nthreads())")
println("-"^72)

resultados = Vector{Any}(undef, nT)
sem = Channel{Nothing}(active_threads)
@sync for (i, T) in enumerate(temperaturas)
    put!(sem, nothing)
    Threads.@spawn begin
        try
            resultados[i] = hysteresis(L, T, B_rama, mcs, n_cycles, master_seed + 100 * i)
            println("[Hilo $(threadid())] T = $T lista")
        finally
            take!(sem)
        end
    end
end

# -------------------------------------------------------------------------
# Tabla, campos coercitivos y area del ciclo
# -------------------------------------------------------------------------
df = DataFrame(T = Float64[], B = Float64[], m_baja = Float64[], m_sube = Float64[])
df_res = DataFrame(T = Float64[], B_coercitivo_baja = Float64[], B_coercitivo_sube = Float64[],
                   m_remanente_baja = Float64[], area_ciclo = Float64[])
println("\nResumen:")
for (i, T) in enumerate(temperaturas)
    m_baja, m_sube, _, _ = resultados[i]
    for k in eachindex(B_rama)
        push!(df, (T, B_rama[k], m_baja[k], m_sube[k]))
    end
    Bc_b = coercive_field(B_rama, m_baja)
    Bc_s = coercive_field(B_rama, m_sube)
    m_rem = m_baja[argmin(abs.(B_rama))]
    # Area encerrada: ∮ m dB = ∫ (m_baja - m_sube) dB sobre la grilla (B decreciente -> signo)
    area = sum((m_baja[1:end - 1] .- m_sube[1:end - 1] .+ m_baja[2:end] .- m_sube[2:end]) ./ 2 .*
               abs.(diff(B_rama)))
    push!(df_res, (T, Bc_b, Bc_s, m_rem, area))
    @printf("  T = %.2f : B_c (baja) = %+.3f, B_c (sube) = %+.3f, m(B=0, baja) = %+.3f, area = %.3f\n",
            T, Bc_b, Bc_s, m_rem, area)
end
CSV.write(joinpath(dir_out, "hysteresis_L$(L).csv"), df)
CSV.write(joinpath(dir_out, "hysteresis_summary_L$(L).csv"), df_res)

# -------------------------------------------------------------------------
# Gráficos
# -------------------------------------------------------------------------
Tc_exacto = 2.269185
paleta = cgrad(:coolwarm, nT, categorical = true)

p_ciclo = plot(xlabel = "B / J", ylabel = "m", legend = :bottomright,
               title = @sprintf("Ciclos de histeresis (L = %d, %d MCS/paso)", L, mcs))
for (i, T) in enumerate(temperaturas)
    m_baja, m_sube, _, _ = resultados[i]
    sufijo = abs(T - Tc_exacto) < 0.01 ? " (≈ Tc)" : (T > Tc_exacto ? " (> Tc)" : "")
    etiqueta = @sprintf("T = %.2f%s", T, sufijo)
    plot!(p_ciclo, B_rama, m_baja, color = paleta[i], lw = 2, label = etiqueta)
    plot!(p_ciclo, B_rama, m_sube, color = paleta[i], lw = 2, linestyle = :dash, label = "")
end
hline!(p_ciclo, [0.0], color = :gray, lw = 0.5, label = "")
vline!(p_ciclo, [0.0], color = :gray, lw = 0.5, label = "")
annotate!(p_ciclo, [(-B_max * 0.95, 0.9, text("— baja   - - sube", 8, :left))])

p_area = plot(df_res.T, df_res.area_ciclo, marker = :circle, lw = 2, color = :black, label = "",
              xlabel = "T", ylabel = "Area del ciclo ∮ m dB", title = "Area vs temperatura")
vline!(p_area, [Tc_exacto], linestyle = :dash, color = :gray, label = "Tc")

# Instantaneas en el punto de m ≈ 0 de la rama descendente
paneles_snap = []
for (i, T) in enumerate(temperaturas)
    _, _, snap, B_snap = resultados[i]
    push!(paneles_snap, heatmap(Float64.(snap), color = cgrad([:navy, :gold]), clims = (-1, 1),
                                aspect_ratio = 1, axis = false, ticks = false, colorbar = false,
                                title = @sprintf("T = %.2f, B = %+.2f", T, B_snap), titlefontsize = 9))
end
fila_snap = plot(paneles_snap..., layout = (1, nT))

fig = plot(plot(p_ciclo, p_area, layout = grid(1, 2, widths = [0.65, 0.35])), fila_snap,
           layout = grid(2, 1, heights = [0.68, 0.32]), size = (1300, 900), margin = 5mm, dpi = 200)
ruta_fig = joinpath(dir_out, "hysteresis_L$(L).png")
savefig(fig, ruta_fig)

duracion_s = round((now() - t_inicio).value / 1000, digits = 1)
println("\nDatos  : $(joinpath(dir_out, "hysteresis_L$(L).csv"))")
println("Figura : $ruta_fig")
println("Duracion: $duracion_s s")
