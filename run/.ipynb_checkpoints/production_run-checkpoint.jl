# run/production_run.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Printf
using Plots
using DataFrames
using CSV
using ArgParse
using Base.Threads
using Statistics

# Cargamos el archivo de simulación base
include(joinpath(@__DIR__, "runising.jl"))

# -------------------------------------------------------------------------
# Argumentos de línea de comandos
# -------------------------------------------------------------------------
function parse_commandline()
    s = ArgParseSettings(description = "Corrida de produccion en equilibrio para Ising 2D con tau adaptativo")

    @add_arg_table! s begin
        "--L"
            help = "Tamano lateral de la red (LxL)"
            arg_type = Int
            default = 16
        "--T_min"
            help = "Temperatura minima"
            arg_type = Float64
            default = 1.8
        "--T_max"
            help = "Temperatura maxima"
            arg_type = Float64
            default = 2.8
        "--nT"
            help = "Numero de temperaturas a evaluar"
            arg_type = Int
            default = 64
        "--n_samples"
            help = "Cantidad de muestras independientes por temperatura"
            arg_type = Int
            default = 2500
        "--threads"
            help = "Numero maximo de hilos a utilizar"
            arg_type = Int
            default = nthreads()
    end

    return parse_args(s)
end

# -------------------------------------------------------------------------
# Carga o modelo del tiempo de autocorrelación adaptativo
# -------------------------------------------------------------------------
function load_or_model_tau(dir_data::String, L::Int)
    # Busca archivos decorrelation_L<L>*.csv en data/
    patron = Regex("^decorrelation_L$(L).*\\.csv\$")
    archivos = filter(f -> occursin(patron, f), readdir(dir_data))

    if !isempty(archivos)
        ruta_csv = joinpath(dir_data, first(archivos))
        println("-> Usando datos de autocorrelacion desde: $ruta_csv")
        df_decorr = CSV.read(ruta_csv, DataFrame)
        sort!(df_decorr, :T)

        return (T::Float64) -> begin
            # Acotado fuera del rango medido
            if T <= df_decorr.T[1]
                tau = df_decorr.tau_int_mag[1]
            elseif T >= df_decorr.T[end]
                tau = df_decorr.tau_int_mag[end]
            else
                idx = findlast(df_decorr.T .<= T)
                T1, T2 = df_decorr.T[idx], df_decorr.T[idx + 1]
                t1, t2 = df_decorr.tau_int_mag[idx], df_decorr.tau_int_mag[idx + 1]
                tau = t1 + (t2 - t1) * (T - T1) / (T2 - T1)
            end
            tau_val = max(1.0, tau)
            sample_interval = max(2, ceil(Int, 2.5 * tau_val))
            n_therm = max(500, ceil(Int, 10.0 * tau_val))
            return sample_interval, n_therm, tau_val
        end
    else
        println("-> No se encontro CSV previo para L=$L en data/. Usando perfil analitico tau ~ L^2.17")
        Tc = 2.269185
        tau_base = 2.0
        # Escalamiento aproximado del pico critico de Glauber
        tau_pico = max(3.0, 0.45 * (L / 10)^2.17)
        ancho = 0.18

        return (T::Float64) -> begin
            tau = tau_base + (tau_pico - tau_base) / (1.0 + ((T - Tc) / ancho)^2)
            sample_interval = max(2, ceil(Int, 2.5 * tau))
            n_therm = max(500, ceil(Int, 10.0 * tau))
            return sample_interval, n_therm, tau
        end
    end
end

# -------------------------------------------------------------------------
# Pipeline de producción
# -------------------------------------------------------------------------
args = parse_commandline()

L = args["L"]
N = L * L
T_min = args["T_min"]
T_max = args["T_max"]
nT = args["nT"]
n_samples = args["n_samples"]
requested_threads = args["threads"]

active_threads = min(requested_threads, nthreads())
J = 1.0
h = 0.0
master_seed = 2026

# Carpeta de salida y entrada
dir_data = joinpath(@__DIR__, "data")
mkpath(dir_data)

# Función adaptativa para (sample_interval, n_therm, tau)
get_adaptive_params = load_or_model_tau(dir_data, L)

temperaturas = collect(range(T_min, T_max, length=nT))

println("="^72)
println(" CORRIDA DE MEDICION EN ESTADO ESTACIONARIO (TAU ADAPTATIVO)")
println("="^72)
println("Hilos en uso: $active_threads / $(nthreads())")
println("Parametros:")
println("  - Red L x L          : $L x $L ($N espines)")
println("  - Puntos de T        : $nT (rango [$T_min, $T_max])")
println("  - Muestras por T     : $n_samples")
println("  - Politica decorr.   : 2.5 * tau_int (adaptativo)")
println("  - Politica therm.    : 10.0 * tau_int (adaptativo)")
println("-"^72)

# Estructuras de almacenamiento thread-safe
avg_m_vec       = zeros(Float64, nT)
err_m_vec       = zeros(Float64, nT)
avg_e_vec       = zeros(Float64, nT)
err_e_vec       = zeros(Float64, nT)
chi_vec         = zeros(Float64, nT)
cv_vec          = zeros(Float64, nT)
binder_vec      = zeros(Float64, nT)
sample_int_vec  = zeros(Int, nT)
n_therm_vec     = zeros(Int, nT)

sem = Channel{Nothing}(active_threads)

@sync for idx in 1:nT
    put!(sem, nothing)
    Threads.@spawn begin
        try
            T = temperaturas[idx]
            rng = Xoshiro(master_seed + idx * 10000)

            # Obtenemos parámetros adaptativos para esta temperatura exacta
            s_interval, n_therm, tau_local = get_adaptive_params(T)
            sample_int_vec[idx] = s_interval
            n_therm_vec[idx] = n_therm

            # Inicializamos red ordenada (Cold Start acelera convergencia)
            s = fill(Int8(1), L, L)

            # 1. Fase de termalización adaptativa
            ising2d!(s, J, h, T, n_therm, rand(rng, UInt64))

            # Vectores temporales para muestras decorrelacionadas
            m_samples = zeros(Float64, n_samples)
            e_samples = zeros(Float64, n_samples)

            # 2. Fase de muestreo estacionario
            for samp in 1:n_samples
                ising2d!(s, J, h, T, s_interval, rand(rng, UInt64))
                m_samples[samp] = magnetization(s)
                e_samples[samp] = energy(s, J, h)
            end

            # 3. Post-procesamiento
            abs_m = abs.(m_samples)
            e_per_spin = e_samples ./ N

            m_mean = mean(abs_m)
            e_mean = mean(e_per_spin)

            err_m = std(abs_m) / sqrt(n_samples)
            err_e = std(e_per_spin) / sqrt(n_samples)

            m2 = mean(m_samples .^ 2)
            m4 = mean(m_samples .^ 4)
            e_total_mean = mean(e_samples)
            e2_total = mean(e_samples .^ 2)

            chi = (N / T) * (m2 - m_mean^2)
            cv = (1 / (N * T^2)) * (e2_total - e_total_mean^2)
            u4 = 1.0 - (m4 / (3.0 * m2^2))

            avg_m_vec[idx]  = m_mean
            err_m_vec[idx]  = err_m
            avg_e_vec[idx]  = e_mean
            err_e_vec[idx]  = err_e
            chi_vec[idx]    = chi
            cv_vec[idx]     = cv
            binder_vec[idx] = u4

            println("[Hilo $(threadid())] T = $(@sprintf("%.3f", T)) | tau ≈ $(@sprintf("%.1f", tau_local)) | dt = $s_interval | Nterm = $n_therm | <|m|> = $(@sprintf("%.4f", m_mean))")
        finally
            take!(sem)
        end
    end
end

# -------------------------------------------------------------------------
# Armado de DataFrame y guardado
# -------------------------------------------------------------------------
df_termo = DataFrame(
    T = temperaturas,
    mag_abs = avg_m_vec,
    mag_err = err_m_vec,
    ene_per_spin = avg_e_vec,
    ene_err = err_e_vec,
    susceptibilidad = chi_vec,
    calor_especifico = cv_vec,
    binder_cumulant = binder_vec,
    sample_interval = sample_int_vec,
    n_therm = n_therm_vec
)

ruta_csv = joinpath(dir_data, "equilibrium_observables_L$(L).csv")
CSV.write(ruta_csv, df_termo)
println("\n" * "="^72)
println("Resultados guardados en: $ruta_csv")

# -------------------------------------------------------------------------
# Generación de gráficos (panel 2x2 de observables + panel de parámetros adaptativos)
# -------------------------------------------------------------------------
println("Generando figuras termodinámicas...")

Tc_exact = 2.269185

p1 = plot(df_termo.T, df_termo.mag_abs, yerror=df_termo.mag_err,
          marker=:circle, lw=1.5, color=:blue, label="<|m|>",
          xlabel="Temperatura T", ylabel="<|m|>", title="Magnetización")
vline!(p1, [Tc_exact], linestyle=:dash, color=:gray, label="Tc Onsager")

p2 = plot(df_termo.T, df_termo.ene_per_spin, yerror=df_termo.ene_err,
          marker=:circle, lw=1.5, color=:green, label="<e>",
          xlabel="Temperatura T", ylabel="Energía por espín", title="Energía Interna")
vline!(p2, [Tc_exact], linestyle=:dash, color=:gray, label="")

p3 = plot(df_termo.T, df_termo.susceptibilidad,
          marker=:square, lw=1.5, color=:red, label="χ",
          xlabel="Temperatura T", ylabel="χ", title="Susceptibilidad Magnética")
vline!(p3, [Tc_exact], linestyle=:dash, color=:gray, label="")

p4 = plot(df_termo.T, df_termo.calor_especifico,
          marker=:diamond, lw=1.5, color=:purple, label="Cv",
          xlabel="Temperatura T", ylabel="Cv", title="Calor Específico")
vline!(p4, [Tc_exact], linestyle=:dash, color=:gray, label="")

fig_termo = plot(p1, p2, p3, p4, layout=(2, 2), size=(900, 700), margin=5Plots.mm)
ruta_fig = joinpath(dir_data, "equilibrium_observables_L$(L).png")
savefig(fig_termo, ruta_fig)
println("Gráficos exportados a: $ruta_fig")