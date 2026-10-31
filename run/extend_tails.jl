# run/extend_tails.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Printf
using Dates
using DataFrames
using CSV
using ArgParse
using Base.Threads
using Statistics

# Simulación base
include(joinpath(@__DIR__, "runising.jl"))

function parse_commandline()
    s = ArgParseSettings(description = "Extension de colas de temperatura con backup para Ising 2D")

    @add_arg_table! s begin
        "--L"
            help = "Tamano lateral de la red"
            arg_type = Int
            required = true
        "--T_min_global"
            help = "Temperatura minima del rango global"
            arg_type = Float64
            default = 1.0
        "--T_max_global"
            help = "Temperatura maxima del rango global"
            arg_type = Float64
            default = 4.0
        "--nT_cola"
            help = "Numero de puntos a agregar por cada cola (baja y alta T)"
            arg_type = Int
            default = 16
        "--n_samples"
            help = "Cantidad de muestras por temperatura"
            arg_type = Int
            default = 3000
        "--threads"
            help = "Hilos a utilizar"
            arg_type = Int
            default = nthreads()
    end

    return parse_args(s)
end

# -------------------------------------------------------------------------
# Pipeline de extensión de colas
# -------------------------------------------------------------------------
t_inicio = now()
args = parse_commandline()

L = args["L"]
N = L * L
T_min_ext = args["T_min_global"]
T_max_ext = args["T_max_global"]
nT_cola = args["nT_cola"]
n_samples = args["n_samples"]
requested_threads = args["threads"]
active_threads = min(requested_threads, nthreads())

dir_data = joinpath(@__DIR__, "data")
ruta_csv = joinpath(dir_data, "equilibrium_observables_L$(L).csv")

if !isfile(ruta_csv)
    error("No se encontro el archivo previo: $ruta_csv. Corre primero production_run.jl para L=$L.")
end

# Lectura de los datos existentes
df_previo = CSV.read(ruta_csv, DataFrame)
sort!(df_previo, :T)

T_min_actual = minimum(df_previo.T)
T_max_actual = maximum(df_previo.T)

# -------------------------------------------------------------------------
# Determinación de temperaturas en las colas exteriores
# -------------------------------------------------------------------------
nuevas_T = Float64[]

# Cola inferior (T < T_min_actual)
if T_min_ext < T_min_actual
    step_low = (T_min_actual - T_min_ext) / nT_cola
    append!(nuevas_T, range(T_min_ext, T_min_actual - step_low, length = nT_cola))
end

# Cola superior (T > T_max_actual)
if T_max_ext > T_max_actual
    step_high = (T_max_ext - T_max_actual) / nT_cola
    append!(nuevas_T, range(T_max_actual + step_high, T_max_ext, length = nT_cola))
end

if isempty(nuevas_T)
    println("El archivo ya cubre o supera el rango [$T_min_ext, $T_max_ext]. Nada por hacer.")
    exit(0)
end

sort!(nuevas_T)
nT_nuevas = length(nuevas_T)

println("="^75)
println(" EXTENSION DE COLAS - ISING 2D (L = $L)")
println("="^75)
println("Fecha y hora de inicio : $(Dates.format(t_inicio, "yyyy-mm-dd HH:MM:SS"))")
println("Rango existente        : [$T_min_actual, $T_max_actual] ($(nrow(df_previo)) puntos)")
println("Rango objetivo         : [$T_min_ext, $T_max_ext]")
println("Puntos nuevos a medir  : $nT_nuevas temperaturas")
println("Hilos en uso           : $active_threads / $(nthreads())")
println("-"^75)

# Valores de referencia de tau tomados de los extremos del CSV existente
tau_borde_inf = max(1.0, df_previo.sample_interval[1] / 2.5)
tau_borde_sup = max(1.0, df_previo.sample_interval[end] / 2.5)

# Contenedores para almacenar las nuevas mediciones
avg_m_vec       = zeros(Float64, nT_nuevas)
err_m_vec       = zeros(Float64, nT_nuevas)
avg_e_vec       = zeros(Float64, nT_nuevas)
err_e_vec       = zeros(Float64, nT_nuevas)
chi_vec         = zeros(Float64, nT_nuevas)
cv_vec          = zeros(Float64, nT_nuevas)
binder_vec      = zeros(Float64, nT_nuevas)
sample_int_vec  = zeros(Int, nT_nuevas)
n_therm_vec     = zeros(Int, nT_nuevas)

J = 1.0
h = 0.0
master_seed = 4096 + L * 31

sem = Channel{Nothing}(active_threads)

@sync for idx in 1:nT_nuevas
    put!(sem, nothing)
    Threads.@spawn begin
        try
            T = nuevas_T[idx]
            rng = Xoshiro(master_seed + idx * 10000)

            # Asignación conservadora del tiempo de autocorrelación desde el borde
            tau_asumido = T < T_min_actual ? tau_borde_inf : tau_borde_sup
            s_interval = max(2, ceil(Int, 2.5 * tau_asumido))
            n_therm    = max(500, ceil(Int, 10.0 * tau_asumido))

            sample_int_vec[idx] = s_interval
            n_therm_vec[idx]    = n_therm

            s = fill(Int8(1), L, L) # Cold start

            # 1. Fase de termalización
            ising2d!(s, J, h, T, n_therm, rand(rng, UInt64))

            # 2. Fase de muestreo
            m_samples = zeros(Float64, n_samples)
            e_samples = zeros(Float64, n_samples)

            for samp in 1:n_samples
                ising2d!(s, J, h, T, s_interval, rand(rng, UInt64))
                m_samples[samp] = magnetization(s)
                e_samples[samp] = energy(s, J, h)
            end

            # 3. Cálculo de observables termodinámicos
            abs_m = abs.(m_samples)
            e_per_spin = e_samples ./ N

            m_mean = mean(abs_m)
            e_mean = mean(e_per_spin)
            err_m  = std(abs_m) / sqrt(n_samples)
            err_e  = std(e_per_spin) / sqrt(n_samples)

            m2 = mean(m_samples .^ 2)
            m4 = mean(m_samples .^ 4)
            e_total_mean = mean(e_samples)
            e2_total     = mean(e_samples .^ 2)

            chi = (N / T) * (m2 - m_mean^2)
            cv  = (1 / (N * T^2)) * (e2_total - e_total_mean^2)
            u4  = 1.0 - (m4 / (3.0 * m2^2))

            avg_m_vec[idx]  = m_mean
            err_m_vec[idx]  = err_m
            avg_e_vec[idx]  = e_mean
            err_e_vec[idx]  = err_e
            chi_vec[idx]    = chi
            cv_vec[idx]     = cv
            binder_vec[idx] = u4

            println("[Hilo $(threadid())] T = $(@sprintf("%.3f", T)) lista | dt = $s_interval | <|m|> = $(@sprintf("%.4f", m_mean))")
        finally
            take!(sem)
        end
    end
end

df_nuevos = DataFrame(
    T = nuevas_T,
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

# Unificación y ordenamiento por temperatura
df_completo = vcat(df_previo, df_nuevos)
sort!(df_completo, :T)

# -------------------------------------------------------------------------
# Escritura Segura: Backup y Reemplazo Atómico
# -------------------------------------------------------------------------
ruta_backup = joinpath(dir_data, "equilibrium_observables_L$(L)_orig_backup.csv")
cp(ruta_csv, ruta_backup; force = true)
println("\n-> Respaldo del archivo original creado en: $ruta_backup")

ruta_tmp = joinpath(dir_data, "equilibrium_observables_L$(L).tmp")
CSV.write(ruta_tmp, df_completo)
mv(ruta_tmp, ruta_csv; force = true)

t_fin = now()
duracion_s = round((t_fin - t_inicio).value / 1000, digits = 2)

println("="^75)
println("Archivo actualizado con exito: $ruta_csv")
println("Total de puntos de temperatura : $(nrow(df_completo)) en [$T_min_ext, $T_max_ext]")
println("Duracion total                 : $(duracion_s) s")
println("="^75)