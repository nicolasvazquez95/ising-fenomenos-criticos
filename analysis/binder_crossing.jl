# analysis/binder_crossing.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using Printf
using Dates
using Random
using Plots, Measures
using DataFrames
using CSV
using ArgParse
using Statistics
using Polynomials

# -------------------------------------------------------------------------
# Argumentos de línea de comandos
# -------------------------------------------------------------------------
function parse_commandline()
    s = ArgParseSettings(description = "Estimacion de Tc por cruce del cumulante de Binder en Ising 2D")

    @add_arg_table! s begin
        "--data_dir"
            help = "Directorio donde residen los CSVs de mediciones (ej. equilibrium_observables_L*.csv)"
            arg_type = String
            default = joinpath(@__DIR__, "..", "run", "data")
        "--out_dir"
            help = "Directorio de destino para graficos y tablas de resultados"
            arg_type = String
            default = joinpath(@__DIR__, "output")
        "--T_center"
            help = "Centro de la ventana de ajuste (estimacion previa de Tc)"
            arg_type = Float64
            default = 2.27
        "--window_a"
            help = "Semiancho de la ventana de ajuste: a / L (escala como L^(-1/nu), nu = 1)"
            arg_type = Float64
            default = 2.0
        "--degree"
            help = "Grado del polinomio local ajustado a U_L(T)"
            arg_type = Int
            default = 3
        "--n_boot"
            help = "Numero de replicas bootstrap parametrico"
            arg_type = Int
            default = 2000
    end

    return parse_args(s)
end

# -------------------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------------------
"""
Ajuste polinomial local de U_L(T) en la ventana |T - T_center| <= a / L.
Devuelve el polinomio, los datos usados y la dispersion de los residuos.
"""
function local_fit(T::Vector{Float64}, U::Vector{Float64}, L::Int;
                   T_center::Float64, a::Float64, degree::Int)
    half_width = a / L
    mask = abs.(T .- T_center) .<= half_width
    T_sub = T[mask]
    U_sub = U[mask]

    if length(T_sub) < degree + 3
        error("L=$L: solo $(length(T_sub)) puntos en la ventana ±$(round(half_width, digits=4)); aumentar --window_a")
    end

    poly = fit(T_sub, U_sub, degree)
    residuals = U_sub .- poly.(T_sub)
    sigma = sqrt(sum(residuals .^ 2) / (length(T_sub) - degree - 1))

    return poly, T_sub, U_sub, sigma, half_width
end

"""
Cruce de dos polinomios dentro del intervalo [T_lo, T_hi]. Si hay varias
raices reales, se toma la mas cercana a T_center. Devuelve NaN si no hay cruce.
"""
function crossing(p1::Polynomial, p2::Polynomial, T_lo::Float64, T_hi::Float64, T_center::Float64)
    raices = roots(p1 - p2)
    candidatos = [real(r) for r in raices if abs(imag(r)) < 1e-10 && T_lo <= real(r) <= T_hi]
    isempty(candidatos) && return NaN
    return candidatos[argmin(abs.(candidatos .- T_center))]
end

"""
Cruce entre dos tamaños con error por bootstrap parametrico: se generan
replicas U* = ajuste + N(0, sigma_residuos) y se recalcula el cruce.
"""
function crossing_with_error(fit1, fit2, T_center::Float64, degree::Int, n_boot::Int, rng)
    p1, T1, U1, s1, w1 = fit1
    p2, T2, U2, s2, w2 = fit2

    # Intervalo comun de validez de ambos ajustes
    T_lo = max(minimum(T1), minimum(T2))
    T_hi = min(maximum(T1), maximum(T2))

    T_x = crossing(p1, p2, T_lo, T_hi, T_center)
    U_x = isnan(T_x) ? NaN : p1(T_x)

    T_boot = Float64[]
    U_boot = Float64[]
    for _ in 1:n_boot
        q1 = fit(T1, p1.(T1) .+ s1 .* randn(rng, length(T1)), degree)
        q2 = fit(T2, p2.(T2) .+ s2 .* randn(rng, length(T2)), degree)
        t = crossing(q1, q2, T_lo, T_hi, T_center)
        if !isnan(t)
            push!(T_boot, t)
            push!(U_boot, q1(t))
        end
    end

    frac_ok = length(T_boot) / n_boot
    T_err = length(T_boot) > 1 ? std(T_boot) : NaN
    U_err = length(U_boot) > 1 ? std(U_boot) : NaN

    return T_x, T_err, U_x, U_err, frac_ok
end

"""
Promedio pesado por 1/sigma^2, ignorando entradas NaN.
"""
function weighted_mean(x::Vector{Float64}, err::Vector{Float64})
    ok = .!isnan.(x) .& .!isnan.(err) .& (err .> 0)
    w = 1.0 ./ err[ok] .^ 2
    mu = sum(w .* x[ok]) / sum(w)
    return mu, sqrt(1.0 / sum(w))
end

# -------------------------------------------------------------------------
# Pipeline de Análisis
# -------------------------------------------------------------------------
args = parse_commandline()

data_dir = abspath(args["data_dir"])
out_dir  = abspath(args["out_dir"])
T_center = args["T_center"]
window_a = args["window_a"]
degree   = args["degree"]
n_boot   = args["n_boot"]
mkpath(out_dir)

Tc_exacto = 2.269185
U_star_ref = 0.61069  # Valor universal del cruce (red cuadrada, PBC; Kamieniarz & Blöte 1993)

println("="^75)
println(" CRUCE DEL CUMULANTE DE BINDER - ISING 2D")
println("="^75)
println("Fecha y hora       : $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("Directorio datos   : $data_dir")
println("Ventana de ajuste  : |T - $T_center| <= $window_a / L, polinomio grado $degree")
println("Replicas bootstrap : $n_boot")
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
println("Tamanos de red encontrados: $L_vec\n")

# -------------------------------------------------------------------------
# 1. Ajustes locales y cruces para todos los pares (L1 < L2)
# -------------------------------------------------------------------------
function compute_crossings(a::Float64, deg::Int, nb::Int; seed::Int = 1234)
    rng = Xoshiro(seed)
    ajustes = Dict(L => local_fit(Float64.(datos_por_L[L].T), Float64.(datos_por_L[L].binder_cumulant), L;
                                  T_center = T_center, a = a, degree = deg) for L in L_vec)
    filas = DataFrame(L1 = Int[], L2 = Int[], T_cross = Float64[], T_cross_err = Float64[],
                      U_cross = Float64[], U_cross_err = Float64[], frac_boot_ok = Float64[])
    for i in 1:length(L_vec), j in (i + 1):length(L_vec)
        L1, L2 = L_vec[i], L_vec[j]
        T_x, T_e, U_x, U_e, f = crossing_with_error(ajustes[L1], ajustes[L2], T_center, deg, nb, rng)
        push!(filas, (L1, L2, T_x, T_e, U_x, U_e, f))
    end
    return filas, ajustes
end

df_cruces, ajustes = compute_crossings(window_a, degree, n_boot)

for L in L_vec
    _, T_sub, _, sigma, w = ajustes[L]
    @printf("L = %-4d : %2d puntos en [%.4f, %.4f], sigma_res(U) = %.2e\n",
            L, length(T_sub), minimum(T_sub), maximum(T_sub), sigma)
end

println("\nCruces T*(L1, L2):")
@printf("%-5s %-5s | %-20s | %-20s | %s\n", "L1", "L2", "T_cross", "U_cross", "boot ok")
println("-"^75)
for r in eachrow(df_cruces)
    @printf("%-5d %-5d | %.5f +/- %.5f   | %.4f +/- %.4f    | %.2f\n",
            r.L1, r.L2, r.T_cross, r.T_cross_err, r.U_cross, r.U_cross_err, r.frac_boot_ok)
end
CSV.write(joinpath(out_dir, "binder_crossings.csv"), df_cruces)

# -------------------------------------------------------------------------
# 2. Estimacion de Tc
# -------------------------------------------------------------------------
# Pares consecutivos (16-32, 32-64, ...): cada tamaño aparece en a lo sumo dos pares
consecutivos = [(L_vec[i], L_vec[i + 1]) for i in 1:length(L_vec) - 1]
es_consec = [(r.L1, r.L2) in consecutivos for r in eachrow(df_cruces)]
df_consec = df_cruces[es_consec, :]

# Se excluye L = 16 del promedio final: correcciones de escala mas grandes
df_final = df_consec[df_consec.L1 .> minimum(L_vec), :]
Tc_binder, Tc_binder_err = weighted_mean(df_final.T_cross, df_final.T_cross_err)
U_star, U_star_err = weighted_mean(df_final.U_cross, df_final.U_cross_err)

# -------------------------------------------------------------------------
# 3. Sistematicos: variacion de ventana y grado del polinomio
# -------------------------------------------------------------------------
println("\nSistematicos (promedio pesado de pares consecutivos con L1 > $(minimum(L_vec))):")
df_sist = DataFrame(window_a = Float64[], degree = Int[], Tc = Float64[], Tc_err = Float64[],
                    U_star = Float64[], U_star_err = Float64[])
for a in (1.5, 2.0, 3.0), deg in (2, 3)
    try
        filas, _ = compute_crossings(a, deg, 500)
        sel = [((r.L1, r.L2) in consecutivos) && r.L1 > minimum(L_vec) for r in eachrow(filas)]
        tc, tce = weighted_mean(filas.T_cross[sel], filas.T_cross_err[sel])
        us, use = weighted_mean(filas.U_cross[sel], filas.U_cross_err[sel])
        push!(df_sist, (a, deg, tc, tce, us, use))
        @printf("  a = %.1f, grado %d : Tc = %.5f +/- %.5f, U* = %.4f +/- %.4f\n", a, deg, tc, tce, us, use)
    catch e
        @printf("  a = %.1f, grado %d : omitido (%s)\n", a, deg, sprint(showerror, e))
    end
end
CSV.write(joinpath(out_dir, "binder_systematics.csv"), df_sist)
Tc_sist = isempty(df_sist) ? NaN : (maximum(df_sist.Tc) - minimum(df_sist.Tc)) / 2

# Estimacion final en formato tabular (la leen otros scripts de analisis)
CSV.write(joinpath(out_dir, "binder_tc_estimate.csv"),
          DataFrame(Tc = [Tc_binder], Tc_err_stat = [Tc_binder_err], Tc_err_sist = [Tc_sist],
                    U_star = [U_star], U_star_err = [U_star_err]))

println("\n" * "="^75)
println(" ESTIMACION DE Tc POR CRUCE DE BINDER")
println("="^75)
@printf(" Tc (Binder)       : %.5f +/- %.5f (estad.) +/- %.5f (sist.)\n", Tc_binder, Tc_binder_err, Tc_sist)
@printf(" Onsager           : %.6f  (desvio = %.1f sigma_estad.)\n", Tc_exacto, (Tc_binder - Tc_exacto) / Tc_binder_err)
@printf(" U* (1 - m4/3m2^2) : %.4f +/- %.4f (referencia %.5f)\n", U_star, U_star_err, U_star_ref)
@printf(" U* guia (m4/m2^2) : %.4f +/- %.4f (referencia %.4f)\n", 3 * (1 - U_star), 3 * U_star_err, 3 * (1 - U_star_ref))
println("="^75)

# -------------------------------------------------------------------------
# 4. Gráficos
# -------------------------------------------------------------------------
println("\nGenerando figuras del cruce de Binder...")
paleta = palette(:thermal, length(L_vec) + 1)

# Panel izquierdo: zoom de U_L(T) con ajustes locales
p_zoom = plot(xlabel = "T", ylabel = "U_L = 1 - <m⁴>/(3<m²>²)", title = "Cruce del cumulante de Binder",
              legend = :bottomleft, xlims = (2.245, 2.295), ylims = (0.45, 0.68))
for (i, L) in enumerate(L_vec)
    df = datos_por_L[L]
    poly, T_sub, _, _, _ = ajustes[L]
    scatter!(p_zoom, df.T, df.binder_cumulant, color = paleta[i], markersize = 3, markerstrokewidth = 0, label = "L = $L")
    T_fino = range(minimum(T_sub), maximum(T_sub), length = 200)
    plot!(p_zoom, T_fino, poly.(T_fino), color = paleta[i], lw = 1.5, label = "")
end
vline!(p_zoom, [Tc_exacto], linestyle = :dash, color = :black, lw = 1, label = "Tc Onsager")
hline!(p_zoom, [U_star_ref], linestyle = :dot, color = :gray, lw = 1, label = "U* universal")

# Panel derecho: T_cross de pares consecutivos vs L2
p_tx = scatter(df_consec.L2, df_consec.T_cross, yerror = df_consec.T_cross_err,
               color = :green, markersize = 5, label = "T*(L, L')",
               xlabel = "L' (tamaño mayor del par)", ylabel = "T de cruce",
               title = "Cruces de pares consecutivos", xscale = :log2, xticks = (df_consec.L2, string.(df_consec.L2)))
hline!(p_tx, [Tc_exacto], linestyle = :dash, color = :black, label = "Tc Onsager")
hspan!(p_tx, [Tc_binder - Tc_binder_err, Tc_binder + Tc_binder_err], color = :green, alpha = 0.2,
       label = @sprintf("Promedio: %.4f", Tc_binder))

# Panel inferior: colapso U_L vs (T - Tc) L^(1/nu), nu = 1 exacto
p_col = plot(xlabel = "(T - Tc) L^{1/ν}", ylabel = "U_L", title = "Colapso de U_L (Tc Binder, ν = 1)",
             legend = :bottomleft, xlims = (-3, 5))
for (i, L) in enumerate(L_vec)
    df = datos_por_L[L]
    plot!(p_col, (df.T .- Tc_binder) .* L, df.binder_cumulant, color = paleta[i], lw = 1.5,
          marker = :circle, markersize = 2, markerstrokewidth = 0, label = "L = $L")
end

fig = plot(p_zoom, p_tx, p_col, layout = @layout([a b; c]), size = (1000, 850), margin = 6mm, dpi = 200)
ruta_fig = joinpath(out_dir, "binder_crossing.png")
savefig(fig, ruta_fig)
println("Gráfico guardado en: $ruta_fig")

# -------------------------------------------------------------------------
# 5. Reporte
# -------------------------------------------------------------------------
ruta_reporte = joinpath(out_dir, "binder_crossing_report.txt")
open(ruta_reporte, "w") do io
    println(io, "="^70)
    println(io, " REPORTE: Tc POR CRUCE DEL CUMULANTE DE BINDER (ISING 2D)")
    println(io, "="^70)
    println(io, "Fecha y hora del analisis : $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io, "Directorio de datos       : $data_dir")
    println(io, "Tamanos L procesados      : $(join(L_vec, ", "))")
    println(io, "Definicion                : U_L = 1 - <m^4> / (3 <m^2>^2)")
    println(io, "Ventana de ajuste         : |T - $T_center| <= $window_a / L, polinomio grado $degree")
    println(io, "Error estadistico         : bootstrap parametrico ($n_boot replicas, residuos del ajuste)")
    println(io, "-"^70)
    println(io, "CRUCES DE PARES CONSECUTIVOS:")
    for r in eachrow(df_consec)
        @printf(io, "  L = %3d / %3d : T* = %.5f +/- %.5f   U* = %.4f +/- %.4f\n",
                r.L1, r.L2, r.T_cross, r.T_cross_err, r.U_cross, r.U_cross_err)
    end
    println(io, "-"^70)
    println(io, "RESULTADO (promedio pesado de pares consecutivos con L1 > $(minimum(L_vec))):")
    @printf(io, "  - Tc (Binder)       : %.5f +/- %.5f (estad.) +/- %.5f (sist.)\n", Tc_binder, Tc_binder_err, Tc_sist)
    @printf(io, "  - Tc Onsager        : %.6f\n", Tc_exacto)
    @printf(io, "  - U* (1 - m4/3m2^2) : %.4f +/- %.4f (referencia %.5f)\n", U_star, U_star_err, U_star_ref)
    @printf(io, "  - U* guia (m4/m2^2) : %.4f +/- %.4f (referencia %.4f)\n", 3 * (1 - U_star), 3 * U_star_err, 3 * (1 - U_star_ref))
    println(io, "="^70)
end
println("Reporte guardado en: $ruta_reporte")
