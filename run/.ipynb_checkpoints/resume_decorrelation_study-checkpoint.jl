# run/resume_decorrelation_study.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Printf
using Dates
using Plots, Measures
using Statistics
using DataFrames
using CSV
using ArgParse
using Base.Threads

include(joinpath(@__DIR__, "runising.jl"))

function parse_commandline()
    s = ArgParseSettings(description = "Reanudacion y extension del estudio de decorrelacion en Ising 2D")

    @add_arg_table! s begin
        "--L"
            help = "Tamano lateral de la red (LxL)"
            arg_type = Int
            required = true
        "--nT"
            help = "Numero de puntos en la malla de temperatura previamente corrida"
            arg_type = Int
            default = 15
        "--add_snapshots"
            help = "Cantidad de snapshots ADICIONALES a simular (0 para solo recalcular correlacion con otro lag)"
            arg_type = Int
            default = 10000
        "--max_lag"
            help = "Nuevo lag maximo para computar rho(t) (por defecto usa 500 o el provisto)"
            arg_type = Int
            default = 500
        "--threads"
            help = "Numero de hilos para el computo"
            arg_type = Int
            default = nthreads()
    end

    return parse_args(s)
end

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

t_inicio = now()
args = parse_commandline()

L = args["L"]
nT = args["nT"]
add_snapshots = args["add_snapshots"]
max_lag = args["max_lag"]
requested_threads = args["threads"]
active_threads = min(requested_threads, nthreads())

dir_data = joinpath(@__DIR__, "data")

ruta_series_mag = joinpath(dir_data, "timeseries_mag_L$(L)_nT$(nT).csv")
ruta_series_ene = joinpath(dir_data, "timeseries_ene_L$(L)_nT$(nT).csv")
ruta_bin        = joinpath(dir_data, "checkpoint_spins_L$(L)_nT$(nT).bin")

if !isfile(ruta_series_mag) || !isfile(ruta_bin)
    error("No se encontraron los checkpoints de la simulación previa para L=$L, nT=$nT en $dir_data.\nAsegurate de haber ejecutado decorrelation_study.jl primero.")
end

# 1. Cargar series temporales existentes
df_prev_mag = CSV.read(ruta_series_mag, DataFrame)
df_prev_ene = CSV.read(ruta_series_ene, DataFrame)
prev_snapshots = nrow(df_prev_mag)

# Obtener temperaturas a partir de las columnas
cols = names(df_prev_mag)
temperaturas = [parse(Float64, replace(c, "T_" => "")) for c in cols]
T_min = minimum(temperaturas)
T_max = maximum(temperaturas)

# 2. Cargar estados de espines finales previos
redes_cargadas = zeros(Int8, L, L, nT)
open(ruta_bin, "r") do io
    read!(io, redes_cargadas)
end

total_snapshots = prev_snapshots + add_snapshots

println("="^72)
println(" REANUDACION DE ESTUDIO DE AUTOCORRELACION - ISING 2D")
println("="^72)
println("Inicio                  : $(Dates.format(t_inicio, "yyyy-mm-dd HH:MM:SS"))")
println("Snapshots previos       : $prev_snapshots")
println("Snapshots adicionales   : $add_snapshots")
println("Snapshots totales acum. : $total_snapshots")
println("Nuevo max_lag           : $max_lag MCS")
println("Hilos en uso            : $active_threads / $(nthreads())")
println("-"^72)

J = 1.0
h = 0.0
mcs_between = 1
master_seed = 98765 + L

tau_m_vec = zeros(Float64, nT)
tau_e_vec = zeros(Float64, nT)
n_term_vec = zeros(Float64, nT)
curvas_rho_mag = [zeros(Float64, max_lag + 1) for _ in 1:nT]

series_mag_totales = [zeros(Float64, total_snapshots) for _ in 1:nT]
series_ene_totales = [zeros(Float64, total_snapshots) for _ in 1:nT]
redes_nuevas = copy(redes_cargadas)

sem = Channel{Nothing}(active_threads)

@sync for idx in 1:nT
    put!(sem, nothing)
    Threads.@spawn begin
        try
            T = temperaturas[idx]
            col_sym = Symbol(cols[idx])

            # Copiamos la serie anterior
            s_mag_prev = df_prev_mag[!, col_sym]
            s_ene_prev = df_prev_ene[!, col_sym]

            # Recuperamos el estado de los espines
            s = copy(redes_cargadas[:, :, idx])

            if add_snapshots > 0
                rng = Xoshiro(master_seed + idx * 1000 + prev_snapshots)
                new_mag = zeros(Float64, add_snapshots)
                new_ene = zeros(Float64, add_snapshots)

                for snap in 1:add_snapshots
                    ising2d!(s, J, h, T, mcs_between, rand(rng, UInt64))
                    new_mag[snap] = magnetization(s)
                    new_ene[snap] = energy(s, J, h)
                end

                redes_nuevas[:, :, idx] .= s
                full_mag = vcat(s_mag_prev, new_mag)
                full_ene = vcat(s_ene_prev, new_ene)
            else
                full_mag = s_mag_prev
                full_ene = s_ene_prev
            end

            series_mag_totales[idx] = full_mag
            series_ene_totales[idx] = full_ene

            descarte = min(5000, div(total_snapshots, 3))
            serie_mag_stat = abs.(full_mag[descarte:end])
            serie_ene_stat = full_ene[descarte:end]

            rho_m = autocorrelation(serie_mag_stat, max_lag)
            rho_e = autocorrelation(serie_ene_stat, max_lag)

            tau_m = integrated_autocorr_time(rho_m) * mcs_between
            tau_e = integrated_autocorr_time(rho_e) * mcs_between

            curvas_rho_mag[idx] = rho_m
            tau_m_vec[idx] = tau_m
            tau_e_vec[idx] = tau_e
            n_term_vec[idx] = 10.0 * max(tau_m, tau_e)

            println("[Hilo $(threadid())] T = $(@sprintf("%.3f", T)) procesada | tau_int(|m|) = $(@sprintf("%.2f", tau_m)) MCS")
        finally
            take!(sem)
        end
    end
end

df_decorr = DataFrame(
    T = temperaturas,
    tau_int_mag = tau_m_vec,
    tau_int_ene = tau_e_vec,
    n_term_sugerido_mcs = n_term_vec
)

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

# Sobrescribir / Actualizar resultados
CSV.write(joinpath(dir_data, "decorrelation_L$(L)_nT$(nT).csv"), df_decorr)

df_rho = DataFrame(lag = collect(0:max_lag))
for idx in 1:nT
    col_name = Symbol(@sprintf("T_%.3f", temperaturas[idx]))
    df_rho[!, col_name] = curvas_rho_mag[idx]
end
CSV.write(joinpath(dir_data, "autocorrelation_curves_L$(L)_nT$(nT).csv"), df_rho)

if add_snapshots > 0
    df_series_mag = DataFrame([Symbol(@sprintf("T_%.3f", temperaturas[i])) => series_mag_totales[i] for i in 1:nT])
    df_series_ene = DataFrame([Symbol(@sprintf("T_%.3f", temperaturas[i])) => series_ene_totales[i] for i in 1:nT])
    CSV.write(ruta_series_mag, df_series_mag)
    CSV.write(ruta_series_ene, df_series_ene)

    open(ruta_bin, "w") do io
        write(io, redes_nuevas)
    end
end

# Regenerar figuras
println("Generando figuras actualizadas...")
lags = 0:max_lag
paleta = cgrad(:coolwarm)

p1 = plot(xlabel = "Lag t (MCS)", ylabel = "Autocorrelacion ρ(t)", 
          title = "Decaimiento temporal |m| (L=$L, snaps=$total_snapshots)", legend = :topright)

for idx in 1:nT
    val_norm = (temperaturas[idx] - T_min) / (T_max - T_min)
    color_curva = paleta[val_norm]
    etiqueta = (idx == 1 || idx == nT) ? "T = $(round(temperaturas[idx], digits=2))" : ""
    
    if idx != idx_pico
        plot!(p1, lags, curvas_rho_mag[idx], color = color_curva, lw = 2, alpha = 0.9, label = etiqueta)
    end
end
plot!(p1, lags, curvas_rho_mag[idx_pico], 
      color = :black, lw = 3, 
      label = "Pico: T = $(round(T_pico, digits=2)) (τ=$(round(tau_pico, digits=1)))")
hline!(p1, [0.0], linestyle = :dash, color = :gray, label = "")

p2 = plot(df_decorr.T, df_decorr.tau_int_mag,
          marker = :circle, lw = 2, label = "τ_int (|m|)",
          xlabel = "Temperatura T", ylabel = "τ_int (MCS)",
          title = "Ralentizacion critica (L=$L)")
plot!(p2, df_decorr.T, df_decorr.tau_int_ene,
      marker = :square, lw = 2, label = "τ_int (E)")
vline!(p2, [2.269], linestyle = :dash, color = :red, label = "Tc exacto (2.269)")
scatter!(p2, [T_pico], [tau_pico], color = :black, markersize = 6, label = "Pico maximo")

fig_final = plot(p1, p2, layout = (1, 2), size = (1000, 450), margin = 5mm, dpi = 200)
ruta_fig = joinpath(dir_data, "decorrelation_L$(L)_nT$(nT).png")
savefig(fig_final, ruta_fig)
println("Graficos guardados en: $ruta_fig")

# Actualizar metadatos
t_fin = now()
duracion_s = round((t_fin - t_inicio).value / 1000, digits = 2)

ruta_meta = joinpath(dir_data, "metadata_L$(L)_nT$(nT).txt")
open(ruta_meta, "w") do io
    println(io, "="^65)
    println(io, " METADATOS DE SIMULACION - ISING 2D (DECORRELACION - REANUDADO)")
    println(io, "="^65)
    println(io, "Fecha y hora de extension: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io, "Duracion de extension    : $(duracion_s) s")
    println(io, "-"^65)
    println(io, "L (tamano de red)        : $L ($(L*L) espines)")
    println(io, "T_min                   : $T_min")
    println(io, "T_max                   : $T_max")
    println(io, "nT (puntos de T)         : $nT")
    println(io, "snapshots previos        : $prev_snapshots")
    println(io, "snapshots agregados      : $add_snapshots")
    println(io, "snapshots totales        : $total_snapshots")
    println(io, "descarte (termalizado)   : $(min(5000, div(total_snapshots, 3)))")
    println(io, "max_lag                 : $max_lag")
    println(io, "mcs_between             : $mcs_between")
    println(io, "-"^65)
    println(io, "Pico maximo tau_int(|m|) : $(round(tau_pico, digits=2)) MCS a T = $(round(T_pico, digits=3))")
    println(io, "="^65)
end

println("Metadatos actualizados en: $ruta_meta")