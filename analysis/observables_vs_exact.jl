# analysis/observables_vs_exact.jl
#
# Item (a) de la Guia 4: curvas de <|m|>, <e>, chi y Cv para todos los L,
# superpuestas con la solucion exacta de red infinita (Onsager / Yang),
# y graficos de <m^2> y <m^4>.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using Printf
using Dates
using Plots, Measures
using DataFrames
using CSV
using ArgParse
using Statistics

# -------------------------------------------------------------------------
# Argumentos de línea de comandos
# -------------------------------------------------------------------------
function parse_commandline()
    s = ArgParseSettings(description = "Observables de Ising 2D vs solucion exacta de red infinita")

    @add_arg_table! s begin
        "--data_dir"
            help = "Directorio donde residen los CSVs de mediciones (ej. equilibrium_observables_L*.csv)"
            arg_type = String
            default = joinpath(@__DIR__, "..", "run", "data")
        "--out_dir"
            help = "Directorio de destino para graficos y tablas de resultados"
            arg_type = String
            default = joinpath(@__DIR__, "output")
        "--dT_exclusion"
            help = "Semiancho alrededor de Tc excluido de la comparacion cuantitativa con la solucion exacta"
            arg_type = Float64
            default = 0.25
    end

    return parse_args(s)
end

# -------------------------------------------------------------------------
# Solucion exacta (J = 1, k_B = 1, por espin)
# -------------------------------------------------------------------------
const Tc_exacto = 2 / log(1 + sqrt(2))

"""
Integrales elipticas completas K(k) y E(k) (modulo k) por media aritmetico-geometrica:
K = π / (2 a_∞),  E = K (1 - Σ_n 2^(n-1) c_n^2), con c_0 = k.
"""
function elliptic_KE(k::Float64)
    a, b, c = 1.0, sqrt(1 - k^2), k
    suma = 0.5 * c^2
    pot = 0.5
    while abs(c) > 1e-15
        a, b, c = (a + b) / 2, sqrt(a * b), (a - b) / 2
        pot *= 2
        suma += pot * c^2
    end
    K = π / (2 * a)
    return K, K * (1 - suma)
end

"""Magnetizacion espontanea de Yang (1952): (1 - sinh(2/T)^-4)^(1/8) para T < Tc."""
mag_exact(T::Float64) = T < Tc_exacto ? (1 - sinh(2 / T)^(-4))^(1 / 8) : 0.0

"""
Energia interna y calor especifico por espin de Onsager (1944), con
κ = 2 sinh(2K) / cosh²(2K) y κ' = 2 tanh²(2K) - 1, K = 1/T.
"""
function onsager_u_c(T::Float64)
    K = 1 / T
    κ = 2 * sinh(2K) / cosh(2K)^2
    κp = 2 * tanh(2K)^2 - 1
    K1, E1 = elliptic_KE(κ)
    u = -coth(2K) * (1 + (2 / π) * κp * K1)
    # Nota: se escribe 2 * E1 porque "2E1" es el literal 2e1 = 20.0 en Julia
    c = (2 / π) * (K * coth(2K))^2 * (2 * K1 - 2 * E1 - (1 - κp) * (π / 2 + κp * K1))
    return u, c
end
ene_exact(T::Float64) = onsager_u_c(T)[1]
cv_exact(T::Float64)  = onsager_u_c(T)[2]

# Chequeos de la implementacion: c = du/dT y limites conocidos
for T in (1.5, 2.0, 2.2, 2.35, 3.0, 4.0)
    h = 1e-5
    c_num = (ene_exact(T + h) - ene_exact(T - h)) / (2h)
    @assert isapprox(c_num, cv_exact(T); rtol = 1e-5) "Cv exacto inconsistente con du/dT en T = $T"
end
@assert isapprox(ene_exact(Tc_exacto + 1e-9), -sqrt(2); atol = 1e-4) "u(Tc) deberia ser -√2"

# -------------------------------------------------------------------------
# Lectura de datos
# -------------------------------------------------------------------------
args = parse_commandline()

data_dir = abspath(args["data_dir"])
out_dir  = abspath(args["out_dir"])
dT_excl  = args["dT_exclusion"]
mkpath(out_dir)

println("="^75)
println(" OBSERVABLES VS SOLUCION EXACTA (ONSAGER / YANG) - ISING 2D")
println("="^75)
println("Fecha y hora       : $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("Directorio datos   : $data_dir")
println("Chequeo Cv = du/dT : OK")
println("-"^75)

patron = Regex("^equilibrium_observables_L(\\d+)\\.csv\$")
archivos = filter(f -> occursin(patron, f), readdir(data_dir))
isempty(archivos) && error("No se encontraron archivos 'equilibrium_observables_L<L>.csv' en $data_dir")

datos_por_L = Dict{Int, DataFrame}()
for arch in archivos
    L_val = parse(Int, match(patron, arch).captures[1])
    df = CSV.read(joinpath(data_dir, arch), DataFrame)
    sort!(df, :T)
    datos_por_L[L_val] = df
end
L_vec = sort(collect(keys(datos_por_L)))
println("Tamanos de red encontrados: $L_vec")
all(L -> "m2" in names(datos_por_L[L]) && "m4" in names(datos_por_L[L]), L_vec) ||
    error("Faltan las columnas m2/m4: correr primero run/append_moments.jl")

# -------------------------------------------------------------------------
# Comparacion cuantitativa lejos de Tc
# -------------------------------------------------------------------------
# <|m|>: solo T < Tc - dT (por encima de Tc la red finita no se anula, item b)
# <e> y Cv: |T - Tc| > dT a ambos lados
df_comp = DataFrame(L = Int[], n_m = Int[], rms_pull_m = Float64[], max_abs_dm = Float64[],
                    n_e = Int[], rms_pull_e = Float64[], max_abs_de = Float64[],
                    n_cv = Int[], mean_rel_dcv = Float64[], rms_rel_dcv = Float64[])
for L in L_vec
    df = datos_por_L[L]
    T = Float64.(df.T)

    sel_m = T .< Tc_exacto - dT_excl
    dm = df.mag_abs[sel_m] .- mag_exact.(T[sel_m])
    pull_m = dm ./ df.mag_err[sel_m]

    sel_e = abs.(T .- Tc_exacto) .> dT_excl
    de = df.ene_per_spin[sel_e] .- ene_exact.(T[sel_e])
    pull_e = de ./ df.ene_err[sel_e]

    cv_ex = cv_exact.(T[sel_e])
    rel_cv = (df.calor_especifico[sel_e] .- cv_ex) ./ cv_ex

    push!(df_comp, (L, count(sel_m), sqrt(mean(pull_m .^ 2)), maximum(abs.(dm)),
                    count(sel_e), sqrt(mean(pull_e .^ 2)), maximum(abs.(de)),
                    count(sel_e), mean(rel_cv), sqrt(mean(rel_cv .^ 2))))
end
CSV.write(joinpath(out_dir, "exact_comparison.csv"), df_comp)

println("\nComparacion con la solucion exacta (|T - Tc| > $dT_excl):")
println("  pull = (simulacion - exacto) / error estadistico; se espera rms ≈ 1")
@printf("  %-5s | %-24s | %-24s | %-22s\n", "L", "<|m|> (T < Tc): rms pull", "<e>: rms pull", "Cv: desvio relativo")
println("  " * "-"^82)
for r in eachrow(df_comp)
    @printf("  %-5d | %5.2f  (max |Δ| %.1e)   | %5.2f  (max |Δ| %.1e)   | %+.2f%% (rms %.2f%%)\n",
            r.L, r.rms_pull_m, r.max_abs_dm, r.rms_pull_e, r.max_abs_de, 100 * r.mean_rel_dcv, 100 * r.rms_rel_dcv)
end

# Valores en Tc de la red finita vs. red infinita (efecto de tamano finito)
println("\nCv maximo de la red finita vs. Cv exacto a la temperatura del maximo:")
for L in L_vec
    df = datos_por_L[L]
    i = argmax(df.calor_especifico)
    @printf("  L = %-4d : Cv_max = %.3f en T = %.4f | exacto en esa T = %.3f\n",
            L, df.calor_especifico[i], df.T[i], cv_exact(Float64(df.T[i])))
end

# -------------------------------------------------------------------------
# Gráficos
# -------------------------------------------------------------------------
println("\nGenerando figuras...")
paleta = palette(:thermal, length(L_vec) + 1)

# Curvas exactas en grilla fina (se evita T = Tc exacto, donde K(κ = 1) diverge)
T_fino = collect(range(1.0, 4.0, length = 3001))
T_fino = T_fino[abs.(T_fino .- Tc_exacto) .> 1e-6]
m_ex = mag_exact.(T_fino)
e_ex = ene_exact.(T_fino)
c_ex = cv_exact.(T_fino)

function panel_observable(col::Symbol, ylab::String, titulo::String; exacto = nothing, kwargs...)
    p = plot(xlabel = "T / J", ylabel = ylab, title = titulo; kwargs...)
    for (i, L) in enumerate(L_vec)
        df = datos_por_L[L]
        plot!(p, df.T, df[!, col], color = paleta[i], lw = 1.5, marker = :circle, markersize = 2,
              markerstrokewidth = 0, label = "L = $L")
    end
    if exacto !== nothing
        plot!(p, T_fino, exacto, color = :black, lw = 2, linestyle = :dash, label = "Exacto (L → ∞)")
    end
    vline!(p, [Tc_exacto], color = :gray, linestyle = :dot, lw = 1, label = "")
    return p
end

# FIGURA 1: rango completo
p_m  = panel_observable(:mag_abs, "<|m|>", "Magnetizacion"; exacto = m_ex, legend = :topright)
p_e  = panel_observable(:ene_per_spin, "<e>", "Energia por espin"; exacto = e_ex, legend = :topleft)
p_chi = panel_observable(:susceptibilidad, "χ", "Susceptibilidad"; legend = :topright, yscale = :log10)
p_cv = panel_observable(:calor_especifico, "Cv", "Calor especifico"; exacto = c_ex, legend = :topright,
                        ylims = (0, 3.2))

fig1 = plot(p_m, p_e, p_chi, p_cv, layout = (2, 2), size = (1100, 850), margin = 6mm, dpi = 200)
savefig(fig1, joinpath(out_dir, "observables_vs_exact.png"))

# FIGURA 2: zoom en la region critica + momentos <m^2> y <m^4>
xz = (2.1, 2.5)
p_mz  = panel_observable(:mag_abs, "<|m|>", "Magnetizacion (zoom)"; exacto = m_ex, legend = :topright, xlims = xz)
p_cvz = panel_observable(:calor_especifico, "Cv", "Calor especifico (zoom)"; exacto = c_ex, legend = :topright,
                         xlims = xz, ylims = (0.5, 3.2))
p_m2  = panel_observable(:m2, "<m²>", "Segundo momento"; exacto = m_ex .^ 2, legend = :topright, xlims = xz)
p_m4  = panel_observable(:m4, "<m⁴>", "Cuarto momento"; exacto = m_ex .^ 4, legend = :topright, xlims = xz)

fig2 = plot(p_mz, p_cvz, p_m2, p_m4, layout = (2, 2), size = (1100, 850), margin = 6mm, dpi = 200)
savefig(fig2, joinpath(out_dir, "observables_near_Tc.png"))
println("Gráficos guardados en: $out_dir")

# -------------------------------------------------------------------------
# Reporte
# -------------------------------------------------------------------------
ruta_reporte = joinpath(out_dir, "observables_vs_exact_report.txt")
open(ruta_reporte, "w") do io
    println(io, "="^78)
    println(io, " REPORTE: OBSERVABLES VS SOLUCION EXACTA DE RED INFINITA (ISING 2D)")
    println(io, "="^78)
    println(io, "Fecha y hora del analisis : $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io, "Directorio de datos       : $data_dir")
    println(io, "Tamanos L procesados      : $(join(L_vec, ", "))")
    println(io, "Solucion exacta           : m de Yang (1952); u y Cv de Onsager (1944)")
    println(io, "Region comparada          : |T - Tc| > $dT_excl (para <|m|>, solo T < Tc - $dT_excl)")
    println(io, "-"^78)
    println(io, "pull = (simulacion - exacto) / error estadistico (se espera rms ≈ 1)")
    for r in eachrow(df_comp)
        @printf(io, "  L = %3d : <|m|> rms pull = %5.2f | <e> rms pull = %5.2f | Cv desvio medio = %+.2f%% (rms %.2f%%)\n",
                r.L, r.rms_pull_m, r.rms_pull_e, 100 * r.mean_rel_dcv, 100 * r.rms_rel_dcv)
    end
    println(io, "="^78)
end
println("Reporte guardado en: $ruta_reporte")
