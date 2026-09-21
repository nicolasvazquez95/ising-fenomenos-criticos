# run/decorrelation_study.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Printf
using Plots, Measures
using Statistics
using DataFrames
using CSV
using ArgParse
using Base.Threads

# Incluimos las funciones del script base
include(joinpath(@__DIR__, "runising.jl"))

# -------------------------------------------------------------------------
# Argumentos por línea de comandos
# -------------------------------------------------------------------------
function parse_commandline()
    s = ArgParseSettings(description = "Estudio de autocorrelacion y termalizacion en Ising 2D")

    @add_arg_table! s begin
        "--L"
            help = "Tamano lateral de la red (LxL)"
            arg_type = Int
            default = 32
        "--T_min"
            help = "Temperatura minima"
            arg_type = Float64
            default = 1.6
        "--T_max"
            help = "Temperatura maxima"
            arg_type = Float64
            default = 3.0
        "--nT"
            help = "Numero de puntos en la malla de temperatura"
            arg_type = Int
            default = 15
        "--snapshots"
            help = "Numero de snapshots por temperatura"
            arg_type = Int
            default = 15000
        "--max_lag"
            help = "Lag maximo para computar rho(t)"
            arg_type = Int
            default = 300
        "--threads"
            help = "Numero maximo de hilos a usar para el computo"
            arg_type = Int
            default = nthreads()
    end

    return parse_args(s)
end

# -------------------------------------------------------------------------
# Funciones de autocorrelación y tiempo integrado
# -------------------------------------------------------------------------
function autocorrelation(x::Vector{Float64}, max_lag::Int)
    N = length(x)
    μ = mean(x)
    var_x = var(x)
    
    if var_x ≈ 0.0
        return zeros(Float64, max_lag + 1)
    end
    
    rho = zeros(Float64, max_lag + 1)
    for lag in 0:max_lag
        acc = 0.0
        for i in 1:(N - lag)
            acc += (x[i] - μ) * (x[i + lag] - μ)
        end
        rho[lag + 1] = (acc / (N - lag)) / var_x
    end
    return rho
end

function integrated_autocorr_time(rho::Vector{Float64}; c::Float64=5.0)
    tau_int = 0.5
    for (k, r) in enumerate(rho[2:end])
        tau_int += r
        if k >= c * tau_int
            return max(0.5, tau_int)
        end
    end
    return max(0.5, tau_int)
end

# -------------------------------------------------------------------------
# Pipeline principal
# -------------------------------------------------------------------------
args = parse_commandline()

L = args["L"]
T_min = args["T_min"]
T_max = args["T_max"]
nT = args["nT"]
n_snapshots = args["snapshots"]
max_lag = args["max_lag"]
requested_threads = args["threads"]

# Verificación del pool de hilos
active_threads = min(requested_threads, nthreads())
if requested_threads > nthreads()
    @warn "Se solicitaron $requested_threads hilos, pero Julia se inicio con $(nthreads()). Ejecuta Julia con 'julia -t $requested_threads ...' para habilitarlos todos. Usando $(nthreads()) hilos."
end

J = 1.0
h = 0.0
mcs_between = 1
master_seed = 42

temperaturas = collect(range(T_min, T_max, length=nT))

println("="^72)
println(" ESTUDIO DE AUTOCORRELACION EN ISING 2D (MULTIHILO)")
println("="^72)
println("Hilos en uso: $active_threads / $(nthreads()) disponibles")
println("Parametros:")
println("  - L            : $L ($(L*L) espines)")
println("  - T min / max  : $T_min / $T_max (puntos: $nT)")
println("  - Snapshots    : $n_snapshots (lag maximo: $max_lag)")
println("-"^72)

tau_m_vec = zeros(Float64, nT)
tau_e_vec = zeros(Float64, nT)
n_term_vec = zeros(Float64, nT)
curvas_rho_mag = [zeros(Float64, max_lag + 1) for _ in 1:nT]

# Canal para restringir la concurrencia exactamente a 'active_threads'
sem = Channel{Nothing}(active_threads)

@sync for idx in 1:nT
    put!(sem, nothing) # Espera turno si se alcanza active_threads
    Threads.@spawn begin
        try
            T = temperaturas[idx]
            seed_thread = master_seed + idx * 1000

            res = run_ising_simulation(L, T, J, h, n_snapshots, seed_thread; mcs_between=mcs_between)

            descarte = min(5000, div(n_snapshots, 3))
            serie_mag = abs.(res[:mag][descarte:end])
            serie_ene = res[:ene][descarte:end]

            rho_m = autocorrelation(serie_mag, max_lag)
            rho_e = autocorrelation(serie_ene, max_lag)

            tau_m = integrated_autocorr_time(rho_m) * mcs_between
            tau_e = integrated_autocorr_time(rho_e) * mcs_between

            curvas_rho_mag[idx] = rho_m
            tau_m_vec[idx] = tau_m
            tau_e_vec[idx] = tau_e
            n_term_vec[idx] = 10.0 * max(tau_m, tau_e)

            println("[Hilo $(threadid())] T = $(@sprintf("%.3f", T)) lista | tau_int(|m|) = $(@sprintf("%.2f", tau_m)) MCS")
        finally
            take!(sem) # Libera el slot para la siguiente tarea
        end
    end
end

df_decorr = DataFrame(
    T = temperaturas,
    tau_int_mag = tau_m_vec,
    tau_int_ene = tau_e_vec,
    n_term_sugerido_mcs = n_term_vec
)

# -------------------------------------------------------------------------
# Identificar la temperatura del pico máximo de tau_int(|m|)
# -------------------------------------------------------------------------
idx_pico = argmax(tau_m_vec)
T_pico = temperaturas[idx_pico]
tau_pico = tau_m_vec[idx_pico]

println("\n" * "="^72)
@printf("%-8s | %-18s | %-18s | %-20s\n", 
        "T", "tau_int (|m|) [MCS]", "tau_int (E) [MCS]", "N_term sugerido [MCS]")
println("-"^72)
for r in eachrow(df_decorr)
    @printf("%-8.3f | %-18.2f | %-18.2f | %-20.0f\n", 
            r.T, r.tau_int_mag, r.tau_int_ene, r.n_term_sugerido_mcs)
end
println("="^72)
println("Pico maximo detectado en T = $(@sprintf("%.3f", T_pico)) con tau_int = $(@sprintf("%.2f", tau_pico)) MCS")

ruta_csv = joinpath(@__DIR__, "data/decorrelation_L$(L)_nT$(nT).csv")
CSV.write(ruta_csv, df_decorr)
println("Resultados exportados a: $ruta_csv")

# -------------------------------------------------------------------------
# Construcción del DataFrame para las curvas de autocorrelación
# -------------------------------------------------------------------------
# Columna base de retardos temporales (lag)
df_rho = DataFrame(lag = collect(0:max_lag))

# Agregamos una columna por cada temperatura simulada
for idx in 1:nT
    # Formato de nombre de columna: ej. T_2.270
    col_name = Symbol(@sprintf("T_%.3f", temperaturas[idx]))
    df_rho[!, col_name] = curvas_rho_mag[idx]
end

# Exportar a CSV
ruta_csv_rho = joinpath(@__DIR__, "data/autocorrelation_curves_L$(L)_nT$(nT).csv")
CSV.write(ruta_csv_rho, df_rho)
println("Curvas de autocorrelación exportadas a: $ruta_csv_rho")

# -------------------------------------------------------------------------
# Graficación con gradiente azul -> rojo y resalte en negro
# -------------------------------------------------------------------------
println("Generando figuras...")
lags = 0:max_lag

# Gradiente continuo de azul (frío) a rojo (caliente)
paleta = cgrad(:coolwarm)

p1 = plot(xlabel = "Lag t (MCS)", ylabel = "Autocorrelacion ρ(t)", 
          title = "Decaimiento temporal |m| (L=$L)", legend = :topright)

# Dibujar todas las curvas con su color del gradiente según su T normalizada
for idx in 1:nT
    # Normalización de T a [0, 1] para la paleta
    val_norm = (temperaturas[idx] - T_min) / (T_max - T_min)
    color_curva = paleta[val_norm]

    # No etiquetamos todas para no saturar la leyenda, solo extremos y pasos
    etiqueta = (idx == 1 || idx == nT) ? "T = $(round(temperaturas[idx], digits=2))" : ""
    
    # Dibujamos las curvas no pico
    if idx != idx_pico
        plot!(p1, lags, curvas_rho_mag[idx], color = color_curva, lw = 2, alpha = 0.9, label = etiqueta)
    end
end

# Dibujar por encima en color NEGRO la curva con el tiempo de correlación máximo
plot!(p1, lags, curvas_rho_mag[idx_pico], 
      color = :black, lw = 3, 
      label = "Pico: T = $(round(T_pico, digits=2)) (τ=$(round(tau_pico, digits=1)))")

hline!(p1, [0.0], linestyle = :dash, color = :gray, label = "")

# Panel 2: Curvas tau_int vs T con el pico marcado
p2 = plot(df_decorr.T, df_decorr.tau_int_mag,
          marker = :circle, lw = 2, label = "τ_int (|m|)",
          xlabel = "Temperatura T", ylabel = "τ_int (MCS)",
          title = "Ralentizacion critica (L=$L)")
plot!(p2, df_decorr.T, df_decorr.tau_int_ene,
      marker = :square, lw = 2, label = "τ_int (E)")
vline!(p2, [2.269], linestyle = :dash, color = :red, label = "Tc exacto (2.269)")
scatter!(p2, [T_pico], [tau_pico], color = :black, markersize = 6, label = "Pico maximo")

fig_final = plot(p1, p2, layout = (1, 2), size = (1000, 450), margin = 5mm, dpi = 200)
ruta_fig = joinpath(@__DIR__, "data/decorrelation_L$(L)_nT$(nT).png")
savefig(fig_final, ruta_fig)
println("Graficos guardados en: $ruta_fig")