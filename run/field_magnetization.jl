# run/field_magnetization.jl
#
# Item (f) de la Guia 4: magnetizacion en funcion del campo B/J para
# temperaturas por encima, por debajo y en la Tc estimada.
#
# Mismo protocolo que production_run.jl, con tau medido punto a punto
# (no hay estudio de decorrelacion previo con campo):
#   1. Termalizacion inicial desde cold start (s = +1, alineado con B > 0)
#   2. Corrida piloto midiendo m en cada MCS para estimar tau_int(m)
#      (se agranda hasta que abarque >= 100 tau)
#   3. Termalizacion adaptativa : max(500, ceil(10 tau))
#   4. Decorrelacion            : max(2, ceil(2.5 tau)) MCS entre muestras
#   5. n_samples muestras; control: tau_int de la serie de muestras (≈ 0.5 si
#      son independientes) y error de la media sigma sqrt(2 tau_muestras / n)
# Cada punto se agrega al CSV al terminar (--resume retoma corridas cortadas).

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Printf
using Dates
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
    s = ArgParseSettings(description = "Magnetizacion vs campo B/J en Ising 2D (item f)")

    @add_arg_table! s begin
        "--L"
            help = "Tamano lateral de la red (LxL)"
            arg_type = Int
            default = 200
        "--Tc"
            help = "Temperatura critica de referencia (por defecto: la estimacion de Binder en analysis/output/binder_tc_estimate.csv)"
            arg_type = Float64
            default = NaN
        "--t_values"
            help = "Temperaturas reducidas t = T/Tc - 1, separadas por comas"
            arg_type = String
            default = "-0.2,-0.1,-0.05,-0.02,-0.01,0,0.01,0.02,0.05,0.1,0.2"
        "--B_min"
            help = "Campo minimo B/J (grilla logaritmica)"
            arg_type = Float64
            default = 1e-3
        "--B_max"
            help = "Campo maximo B/J"
            arg_type = Float64
            default = 1.0
        "--nB"
            help = "Numero de campos"
            arg_type = Int
            default = 19
        "--B_extra"
            help = "Campos adicionales fuera de la grilla logaritmica, separados por comas (vacio: ninguno)"
            arg_type = String
            default = "2,5"
        "--n_samples"
            help = "Cantidad de muestras independientes por punto (T, B)"
            arg_type = Int
            default = 3000
        "--n_burn"
            help = "Termalizacion inicial previa a la corrida piloto [MCS]"
            arg_type = Int
            default = 500
        "--n_pilot"
            help = "Longitud inicial de la corrida piloto [MCS]"
            arg_type = Int
            default = 5000
        "--n_pilot_max"
            help = "Longitud maxima de la corrida piloto [MCS]"
            arg_type = Int
            default = 2_000_000
        "--out_dir"
            help = "Directorio de salida"
            arg_type = String
            default = joinpath(@__DIR__, "data")
        "--resume"
            help = "Retomar: saltear los puntos (T, B) ya presentes en el CSV de salida"
            action = :store_true
        "--threads"
            help = "Numero maximo de hilos a utilizar"
            arg_type = Int
            default = nthreads()
    end

    return parse_args(s)
end

# -------------------------------------------------------------------------
# Tiempo de autocorrelacion integrado (ventana autoconsistente de Sokal)
# -------------------------------------------------------------------------
"""
Devuelve (tau_int, ventana_ok). El sumatorio de rho(t) se corta en el primer
lag t >= c * tau_int; ventana_ok = false si eso no ocurre antes de n/4.
"""
function tau_int(x::Vector{Float64}; c::Float64 = 5.0)
    n = length(x)
    μ = mean(x)
    v = sum((x .- μ) .^ 2) / n
    v <= 1e-300 && return 0.5, true  # serie constante (campo fuerte, T baja)

    τ = 0.5
    for lag in 1:(n ÷ 4)
        acc = 0.0
        @inbounds for i in 1:(n - lag)
            acc += (x[i] - μ) * (x[i + lag] - μ)
        end
        τ += acc / (n - lag) / v
        lag >= c * τ && return max(0.5, τ), true
    end
    return max(0.5, τ), false
end

# -------------------------------------------------------------------------
# Simulacion de un punto (T, B)
# -------------------------------------------------------------------------
function simulate_point(L::Int, T::Float64, B::Float64, seed::Integer;
                        n_samples::Int, n_burn::Int, n_pilot0::Int, n_pilot_max::Int)
    J = 1.0
    N = L * L
    rng = Xoshiro(seed)
    s = fill(Int8(1), L, L)  # Cold start, alineado con B > 0
    sweep!(n) = ising2d!(s, J, B, T, n, rand(rng, UInt64))

    # 1. Termalizacion inicial
    sweep!(n_burn)
    n_therm = n_burn

    # 2. Piloto con paso 1 MCS; se cuadruplica hasta abarcar >= 100 tau
    n_pilot = n_pilot0
    tau = 0.5
    pilot_ok = false
    while true
        m_p = Vector{Float64}(undef, n_pilot)
        for k in 1:n_pilot
            sweep!(1)
            m_p[k] = magnetization(s)
        end
        n_therm += n_pilot
        tau, ventana_ok = tau_int(m_p)
        pilot_ok = ventana_ok && n_pilot >= 100 * tau
        (pilot_ok || n_pilot >= n_pilot_max) && break
        n_pilot = min(4 * n_pilot, n_pilot_max)
    end
    tau_val = max(1.0, tau)

    # 3. Termalizacion adaptativa (misma politica que production_run.jl)
    n_therm_adapt = max(500, ceil(Int, 10.0 * tau_val))
    sweep!(n_therm_adapt)
    n_therm += n_therm_adapt

    # 4. Muestreo con decorrelacion adaptativa
    sample_interval = max(2, ceil(Int, 2.5 * tau_val))
    m_samples = zeros(Float64, n_samples)
    e_samples = zeros(Float64, n_samples)
    for k in 1:n_samples
        sweep!(sample_interval)
        m_samples[k] = magnetization(s)
        e_samples[k] = energy(s, J, B) / N
    end

    # 5. Estadistica y control de independencia de las muestras
    #    (tau_samples ≈ 0.5 si son independientes; si no, el error se corrige)
    tau_s_m, _ = tau_int(m_samples)
    tau_s_e, _ = tau_int(e_samples)

    m_mean = mean(m_samples)
    e_mean = mean(e_samples)
    m_err = std(m_samples) * sqrt(2 * tau_s_m / n_samples)
    e_err = std(e_samples) * sqrt(2 * tau_s_e / n_samples)
    chi = (N / T) * (mean(m_samples .^ 2) - m_mean^2)

    ok = pilot_ok && tau_s_m < 1.0

    return (mag = m_mean, mag_err = m_err, mag_abs = mean(abs.(m_samples)),
            ene_per_spin = e_mean, ene_err = e_err, chi = chi,
            tau_int_mag = tau, tau_samples = tau_s_m,
            sample_interval = sample_interval, n_therm = n_therm,
            pilot_ok = pilot_ok, ok = ok)
end

# -------------------------------------------------------------------------
# Pipeline principal
# -------------------------------------------------------------------------
args = parse_commandline()
t_inicio = now()

L = args["L"]
t_vals = parse.(Float64, strip.(split(args["t_values"], ",")))
B_min, B_max, nB = args["B_min"], args["B_max"], args["nB"]
n_samples = args["n_samples"]
n_burn = args["n_burn"]
n_pilot = args["n_pilot"]
n_pilot_max = args["n_pilot_max"]
dir_out = abspath(args["out_dir"])
requested_threads = args["threads"]
active_threads = min(requested_threads, nthreads())
master_seed = 2028
mkpath(dir_out)

# Tc de referencia: argumento explicito o estimacion por cruce de Binder
Tc = args["Tc"]
origen_Tc = "argumento --Tc"
if isnan(Tc)
    ruta_binder = joinpath(@__DIR__, "..", "analysis", "output", "binder_tc_estimate.csv")
    isfile(ruta_binder) || error("No se encontro $ruta_binder: correr analysis/binder_crossing.jl o pasar --Tc")
    Tc = CSV.read(ruta_binder, DataFrame).Tc[1]
    origen_Tc = "cruce de Binder ($(basename(ruta_binder)))"
end

temperaturas = Tc .* (1 .+ t_vals)
campos_log = nB == 1 ? [B_min] : exp10.(range(log10(B_min), log10(B_max), length = nB))
# Los campos extra van al final para no alterar las semillas (indice iB) de la grilla logaritmica
B_extra = isempty(strip(args["B_extra"])) ? Float64[] : parse.(Float64, strip.(split(args["B_extra"], ",")))
campos = vcat(campos_log, filter(B -> !any(isapprox.(B, campos_log; rtol = 1e-12)), B_extra))

# Tareas: se lanzan primero las mas costosas (|t| chico, B chico) para balancear hilos
tareas = vec([(iT, iB) for iT in eachindex(t_vals), iB in eachindex(campos)])
sort!(tareas, by = x -> (abs(t_vals[x[1]]), campos[x[2]]))

ruta_csv = joinpath(dir_out, "field_magnetization_L$(L).csv")
columnas = ["L", "T", "t", "Tc_used", "B", "mag", "mag_err", "mag_abs", "ene_per_spin", "ene_err", "chi",
            "tau_int_mag", "tau_samples", "sample_interval", "n_therm", "n_samples", "pilot_ok", "ok", "seconds"]

# Reanudacion: se omiten los puntos (T, B) ya guardados
hechos = Set{Tuple{Float64, Float64}}()
if isfile(ruta_csv)
    args["resume"] || error("Ya existe $ruta_csv. Usar --resume para continuarlo o moverlo/borrarlo para empezar de cero.")
    df_prev = CSV.read(ruta_csv, DataFrame)
    for r in eachrow(df_prev)
        push!(hechos, (round(r.T, sigdigits = 10), round(r.B, sigdigits = 10)))
    end
else
    open(io -> println(io, join(columnas, ",")), ruta_csv, "w")
end
filter!(x -> !((round(temperaturas[x[1]], sigdigits = 10), round(campos[x[2]], sigdigits = 10)) in hechos), tareas)

println("="^78)
println(" MAGNETIZACION VS CAMPO - ISING 2D (ITEM f)")
println("="^78)
println("Fecha y hora de inicio : $(Dates.format(t_inicio, "yyyy-mm-dd HH:MM:SS"))")
println("Hilos en uso           : $active_threads / $(nthreads())")
println("Parametros:")
println("  - Red L x L          : $L x $L ($(L * L) espines)")
@printf("  - Tc de referencia   : %.6f (%s)\n", Tc, origen_Tc)
println("  - t = T/Tc - 1       : $(join(t_vals, ", "))")
@printf("  - Campos B/J         : %d valores log-espaciados en [%.1e, %.1e]\n", length(campos_log), B_min, B_max)
isempty(B_extra) || println("  - Campos extra B/J   : $(join(B_extra, ", "))")
println("  - Muestras por punto : $n_samples")
println("  - Politica decorr.   : max(2, ceil(2.5 * tau)) (tau por corrida piloto)")
println("  - Politica therm.    : max(500, ceil(10.0 * tau))")
println("  - Puntos a simular   : $(length(tareas)) (ya hechos: $(length(hechos)))")
println("  - Salida             : $ruta_csv")
println("-"^78)

lock_csv = ReentrantLock()
n_hechos = Atomic{Int}(0)
n_total = length(tareas)
n_dudosos = Atomic{Int}(0)
sem = Channel{Nothing}(active_threads)

@sync for (iT, iB) in tareas
    put!(sem, nothing)
    Threads.@spawn begin
        try
            T, t, B = temperaturas[iT], t_vals[iT], campos[iB]
            seed = master_seed + 10000 * iT + iB
            t0 = time()
            r = simulate_point(L, T, B, seed; n_samples = n_samples, n_burn = n_burn,
                               n_pilot0 = n_pilot, n_pilot_max = n_pilot_max)
            dt = time() - t0
            r.ok || atomic_add!(n_dudosos, 1)

            fila = (L, T, t, Tc, B, r.mag, r.mag_err, r.mag_abs, r.ene_per_spin, r.ene_err, r.chi,
                    r.tau_int_mag, r.tau_samples, r.sample_interval, r.n_therm, n_samples,
                    r.pilot_ok, r.ok, round(dt, digits = 1))
            lock(lock_csv) do
                open(io -> println(io, join(string.(fila), ",")), ruta_csv, "a")
            end

            k = atomic_add!(n_hechos, 1) + 1
            @printf("[%3d/%3d | hilo %2d] t = %+.3f B = %.2e | m = %.5f ± %.1e | tau = %7.1f | dt = %5d | tau_muestras = %.2f | %s %.0f s\n",
                    k, n_total, threadid(), t, B, r.mag, r.mag_err, r.tau_int_mag, r.sample_interval,
                    r.tau_samples, r.ok ? "ok" : "REVISAR", dt)
        finally
            take!(sem)
        end
    end
end

t_fin = now()
duracion_s = round((t_fin - t_inicio).value / 1000, digits = 2)

# -------------------------------------------------------------------------
# Metadatos
# -------------------------------------------------------------------------
df_final = CSV.read(ruta_csv, DataFrame)
ruta_meta = joinpath(dir_out, "metadata_field_L$(L).txt")
open(ruta_meta, "w") do io
    println(io, "="^65)
    println(io, " METADATOS DE SIMULACION - ISING 2D (MAGNETIZACION VS CAMPO)")
    println(io, "="^65)
    println(io, "Fecha y hora de inicio   : $(Dates.format(t_inicio, "yyyy-mm-dd HH:MM:SS"))")
    println(io, "Fecha y hora de fin      : $(Dates.format(t_fin, "yyyy-mm-dd HH:MM:SS"))")
    println(io, "Tiempo de esta sesion    : $(duracion_s) s ($(n_total) puntos)")
    println(io, "-"^65)
    println(io, "L (tamano de red)           : $L ($(L * L) espines)")
    @printf(io, "Tc de referencia            : %.6f (%s)\n", Tc, origen_Tc)
    println(io, "t = T/Tc - 1                : $(join(t_vals, ", "))")
    @printf(io, "Campos B/J                  : %d log-espaciados en [%.1e, %.1e]\n", length(campos_log), B_min, B_max)
    println(io, "Campos extra B/J            : $(isempty(B_extra) ? "ninguno" : join(B_extra, ", "))")
    println(io, "n_samples (por punto)       : $n_samples")
    println(io, "n_burn / n_pilot (inicial)  : $n_burn / $n_pilot")
    println(io, "n_pilot_max                 : $n_pilot_max")
    println(io, "master_seed                 : $master_seed")
    println(io, "J                           : 1.0")
    println(io, "Politica decorrelacion (dt) : max(2, ceil(2.5 * tau))")
    println(io, "Politica termalizacion      : max(500, ceil(10.0 * tau)) (despues de burn-in + piloto)")
    println(io, "Estimacion de tau           : corrida piloto por punto (>= 100 tau)")
    println(io, "Condicion inicial           : Cold start (s = +1)")
    println(io, "Hilos solicitados           : $requested_threads")
    println(io, "Hilos efectivos usados      : $active_threads")
    println(io, "-"^65)
    println(io, "Puntos totales en CSV       : $(nrow(df_final))")
    println(io, "Puntos a revisar (ok=false) : $(count(.!df_final.ok))")
    println(io, "Rango tau_int(m) [MCS]      : [$(round(minimum(df_final.tau_int_mag), digits = 1)), $(round(maximum(df_final.tau_int_mag), digits = 1))]")
    println(io, "Rango sample_interval [MCS] : [$(minimum(df_final.sample_interval)), $(maximum(df_final.sample_interval))]")
    println(io, "="^65)
end

println("\n" * "="^78)
println("Resultados en        : $ruta_csv")
println("Metadatos en         : $ruta_meta")
println("Duracion de sesion   : $(duracion_s) s")
println("Puntos a revisar     : $(n_dudosos[]) de $n_total en esta sesion")
println("="^78)
