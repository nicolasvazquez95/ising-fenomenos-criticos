# analysis/critical_analysis.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using Printf
using Dates
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
    s = ArgParseSettings(description = "Analisis de tamano finito y exponentes criticos para Ising 2D")

    @add_arg_table! s begin
        "--data_dir"
            help = "Directorio donde residen los CSVs de mediciones (ej. equilibrium_observables_L*.csv)"
            arg_type = String
            default = joinpath(@__DIR__, "..", "run", "data")
        "--out_dir"
            help = "Directorio de destino para graficos y tablas de resultados"
            arg_type = String
            default = joinpath(@__DIR__, "output")
        "--window_points"
            help = "Puntos vecinos alrededor del maximo discreto para interpolacion parabolica"
            arg_type = Int
            default = 3
    end

    return parse_args(s)
end

# -------------------------------------------------------------------------
# Funciones auxiliares para picos y regresiones
# -------------------------------------------------------------------------
"""
Ajusta una parábola local alrededor del máximo discreto para hallar con mayor
resolución el pico (T_peak, Y_peak) y su ancho a media altura (FWHM).
"""
function refine_peak(T::Vector{Float64}, Y::Vector{Float64}; half_window::Int = 3)
    idx_max = argmax(Y)
    i_start = max(1, idx_max - half_window)
    i_end   = min(length(T), idx_max + half_window)

    T_sub = T[i_start:i_end]
    Y_sub = Y[i_start:i_end]

    # Ajuste cuadrático local: Y = a*T^2 + b*T + c
    poly = fit(T_sub, Y_sub, 2)
    c, b, a = coeffs(poly)

    if a < 0.0
        T_peak = -b / (2.0 * a)
        Y_peak = poly(T_peak)
    else
        T_peak = T[idx_max]
        Y_peak = Y[idx_max]
    end

    # Estimación de ancho a media altura (FWHM)
    half_max = Y_peak / 2.0
    # Cruce por la izquierda
    idx_left = findlast(Y[1:idx_max] .<= half_max)
    T_left = idx_left === nothing ? T[1] : T[idx_left]
    # Cruce por la derecha
    idx_right_rel = findfirst(Y[idx_max:end] .<= half_max)
    T_right = idx_right_rel === nothing ? T[end] : T[idx_max + idx_right_rel - 1]

    width = abs(T_right - T_left)

    return T_peak, Y_peak, width
end

"""
Regresión lineal simple por cuadrados mínimos y cálculo de R², error de pendiente
y error de ordenada al origen.
"""
function linear_fit(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    x_mean = mean(x)
    y_mean = mean(y)

    ss_xx = sum((x .- x_mean).^2)
    ss_xy = sum((x .- x_mean) .* (y .- y_mean))
    ss_yy = sum((y .- y_mean).^2)

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    y_pred = intercept .+ slope .* x
    residuals = y .- y_pred
    s_err = sqrt(sum(residuals.^2) / max(1, n - 2))
    slope_err = s_err / sqrt(ss_xx)
    intercept_err = s_err * sqrt(1.0 / n + x_mean^2 / ss_xx)

    r2 = (ss_xy^2) / (ss_xx * ss_yy)
    return slope, intercept, slope_err, r2, intercept_err
end

# -------------------------------------------------------------------------
# Pipeline de Análisis
# -------------------------------------------------------------------------
args = parse_commandline()

data_dir = abspath(args["data_dir"])
out_dir  = abspath(args["out_dir"])
w_points = args["window_points"]
mkpath(out_dir)

println("="^75)
println(" ANALISIS DE ESCALAMIENTO DE TAMANO FINITO (FSS) - ISING 2D")
println("="^75)
println("Fecha y hora       : $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("Directorio datos   : $data_dir")
println("Directorio salida  : $out_dir")
println("-"^75)

# 1. Identificar archivos disponibles: equilibrium_observables_L<L>.csv
patron = Regex("^equilibrium_observables_L(\\d+)\\.csv\$")
archivos = filter(f -> occursin(patron, f), readdir(data_dir))

if isempty(archivos)
    error("No se encontraron archivos con formato 'equilibrium_observables_L<L>.csv' en $data_dir")
end

# Extraer y ordenar por L ascendente
datos_por_L = Dict{Int, DataFrame}()
for arch in archivos
    m = match(patron, arch)
    L_val = parse(Int, m.captures[1])
    df = CSV.read(joinpath(data_dir, arch), DataFrame)
    sort!(df, :T)
    datos_por_L[L_val] = df
end

L_vec = sort(collect(keys(datos_por_L)))
println("Tamanos de red encontrados: $L_vec\n")

# -------------------------------------------------------------------------
# 2. Extracción de Picos y Anchos
# -------------------------------------------------------------------------
T_star_chi = Float64[]
chi_max_vec = Float64[]
width_chi_vec = Float64[]

T_star_cv = Float64[]
cv_max_vec = Float64[]
width_cv_vec = Float64[]

for L in L_vec
    df = datos_por_L[L]

    # Pico de Susceptibilidad
    t_chi, max_chi, w_chi = refine_peak(df.T, df.susceptibilidad; half_window = w_points)
    push!(T_star_chi, t_chi)
    push!(chi_max_vec, max_chi)
    push!(width_chi_vec, w_chi)

    # Pico de Calor Específico
    t_cv, max_cv, w_cv = refine_peak(df.T, df.calor_especifico; half_window = w_points)
    push!(T_star_cv, t_cv)
    push!(cv_max_vec, max_cv)
    push!(width_cv_vec, w_cv)
end

df_picos = DataFrame(
    L = L_vec,
    T_star_chi = T_star_chi,
    chi_max = chi_max_vec,
    FWHM_chi = width_chi_vec,
    T_star_cv = T_star_cv,
    cv_max = cv_max_vec,
    FWHM_cv = width_cv_vec
)

println("Resumen de picos termodinamicos por L:")
println(df_picos)
CSV.write(joinpath(out_dir, "fss_peaks_summary.csv"), df_picos)

# -------------------------------------------------------------------------
# 3. Estimación de Exponentes Críticos
# -------------------------------------------------------------------------
ln_L = log.(Float64.(L_vec))
inv_L = 1.0 ./ Float64.(L_vec)
Tc_exacto = 2.269185

# A) Exponente gamma / nu: max(chi) ~ L^(gamma / nu)
slope_chi, int_chi, err_chi, r2_chi = linear_fit(ln_L, log.(chi_max_vec))

# B) Escalamiento del ancho del pico: FWHM ~ L^(-1/nu)
slope_w_chi, int_w_chi, err_w_chi, r2_w_chi = linear_fit(ln_L, log.(width_chi_vec))
nu_est_width = -1.0 / slope_w_chi

# C) Temperatura critica extrapolada de T*(L) vs 1/L
slope_Tc_ext, int_Tc_ext, _, r2_Tc_ext, err_Tc_ext = linear_fit(inv_L, T_star_chi)
Tc_extrapolada = int_Tc_ext

# D) Calor específico: max(Cv) ~ L^(alpha/nu) (en 2D alpha=0, divergen logarítmicamente)
slope_cv, int_cv, err_cv, r2_cv = linear_fit(ln_L, log.(cv_max_vec))

println("\n" * "="^75)
println(" ESTIMACION NUMERICA DE EXPONENTES CRITICOS")
println("="^75)
@printf(" gamma / nu (teorico = 1.750)  : %.4f +/- %.4f (R² = %.4f)\n", slope_chi, err_chi, r2_chi)
@printf(" 1 / nu por ancho (teorico = 1.000) : %.4f +/- %.4f -> nu ≈ %.3f\n", -slope_w_chi, err_w_chi, nu_est_width)
@printf(" Tc extrapolado 1/L -> 0 (Onsager = 2.269): %.4f +/- %.4f\n", Tc_extrapolada, err_Tc_ext)
@printf(" alpha / nu aparente (teorico = 0): %.4f +/- %.4f (R² = %.4f)\n", slope_cv, err_cv, r2_cv)
println("="^75)

# -------------------------------------------------------------------------
# 4. Generación de Gráficos
# -------------------------------------------------------------------------
println("\nGenerando figuras de conjunto y ajustes FSS...")
paleta = palette(:thermal, length(L_vec))

# FIGURA 1: Panel 2x2 con todas las curvas superpuestas
p_m = plot(xlabel="T", ylabel="<|m|>", title="Magnetizacion", legend=:topright)
p_chi = plot(xlabel="T", ylabel="χ", title="Susceptibilidad Magnetica", legend=false)
p_cv = plot(xlabel="T", ylabel="Cv", title="Calor Especifico", legend=false)
p_binder = plot(xlabel="T", ylabel="Cumulante U_L", title="Cumulante de Binder (Cruce)", legend=:bottomleft)

for (i, L) in enumerate(L_vec)
    df = datos_por_L[L]
    plot!(p_m, df.T, df.mag_abs, lw=1.8, color=paleta[i], label="L = $L")
    plot!(p_chi, df.T, df.susceptibilidad, lw=1.8, color=paleta[i], label="L = $L")
    plot!(p_cv, df.T, df.calor_especifico, lw=1.8, color=paleta[i], label="L = $L")
    plot!(p_binder, df.T, df.binder_cumulant, lw=1.8, color=paleta[i], label="L = $L")
end

# Líneas verticales de referencia en Tc exacto
for p in (p_m, p_chi, p_cv, p_binder)
    vline!(p, [Tc_exacto], linestyle=:dash, color=:black, lw=1, label="")
end

fig_all_curves = plot(p_m, p_chi, p_cv, p_binder, layout=(2,2), size=(1000, 800), margin=6mm, dpi=200)
savefig(fig_all_curves, joinpath(out_dir, "all_observables_curves.png"))

# FIGURA 2: Panel de ajustes de escala de tamaño finito (FSS)
# Ajuste chi_max vs L
p_fit_chi = scatter(ln_L, log.(chi_max_vec), color=:red, markersize=6, label="Datos max(χ)")
plot!(p_fit_chi, ln_L, int_chi .+ slope_chi .* ln_L, color=:black, lw=2, 
      label=@sprintf("Ajuste: γ/ν = %.3f", slope_chi),
      xlabel="ln(L)", ylabel="ln(χ_max)", title="Escalamiento χ_max ~ L^(γ/ν)")

# Ajuste FWHM vs L
p_fit_w = scatter(ln_L, log.(width_chi_vec), color=:blue, markersize=6, label="Ancho FWHM(χ)")
plot!(p_fit_w, ln_L, int_w_chi .+ slope_w_chi .* ln_L, color=:black, lw=2,
      label=@sprintf("Pendiente = %.3f (ν≈%.2f)", slope_w_chi, nu_est_width),
      xlabel="ln(L)", ylabel="ln(Ancho)", title="Ancho del pico ~ L^(-1/ν)")

# Extrapolación de Tc
p_fit_tc = scatter(inv_L, T_star_chi, color=:green, markersize=6, label="T*(L) χ")
plot!(p_fit_tc, [0.0; inv_L], int_Tc_ext .+ slope_Tc_ext .* [0.0; inv_L], color=:black, lw=2,
      label=@sprintf("Extrapolado Tc = %.3f", Tc_extrapolada),
      xlabel="1/L", ylabel="T*(L)", title="Extrapolacion de Tc (1/L -> 0)")
hline!(p_fit_tc, [Tc_exacto], linestyle=:dash, color=:gray, label="Tc Onsager")

# Ajuste Cv_max vs ln(L)
p_fit_cv = scatter(ln_L, cv_max_vec, color=:purple, markersize=6, label="max(Cv)")
plot!(p_fit_cv, ln_L, cv_max_vec, lw=1.5, linestyle=:dot, color=:purple, label="",
      xlabel="ln(L)", ylabel="Cv_max", title="Crecimiento max(Cv) vs ln(L)")

fig_fss_fits = plot(p_fit_chi, p_fit_w, p_fit_tc, p_fit_cv, layout=(2,2), size=(1000, 800), margin=6mm, dpi=200)
savefig(fig_fss_fits, joinpath(out_dir, "fss_scaling_fits.png"))

# -------------------------------------------------------------------------
# 5. Exportar reporte de metadatos del análisis
# -------------------------------------------------------------------------
ruta_reporte = joinpath(out_dir, "critical_exponents_report.txt")
open(ruta_reporte, "w") do io
    println(io, "="^70)
    println(io, " REPORTE DE EXPONENTES CRITICOS Y ESCALAMIENTO FSS (ISING 2D)")
    println(io, "="^70)
    println(io, "Fecha y hora del analisis : $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io, "Directorio de datos       : $data_dir")
    println(io, "Tamanos L procesados      : $(join(L_vec, ", "))")
    println(io, "-"^70)
    println(io, "RESULTADOS DE AJUSTE:")
    @printf(io, "  - gamma / nu            : %.4f +/- %.4f (Exacto: 1.750) [R2 = %.4f]\n", slope_chi, err_chi, r2_chi)
    @printf(io, "  - 1 / nu (ancho del pico): %.4f +/- %.4f (Exacto: 1.000) -> nu ≈ %.3f\n", -slope_w_chi, err_w_chi, nu_est_width)
    @printf(io, "  - Tc extrapolado        : %.4f +/- %.4f (Onsager: 2.269185)\n", Tc_extrapolada, err_Tc_ext)
    @printf(io, "  - alpha / nu (Cv)       : %.4f +/- %.4f (Exacto: 0.0 log) [R2 = %.4f]\n", slope_cv, err_cv, r2_cv)
    println(io, "="^70)
end

println("Archivos generados con exito en: $out_dir")

# -------------------------------------------------------------------------
# FIGURA 3: Gráfico de Colapso de Curvas (Data Collapse)
# -------------------------------------------------------------------------
println("Generando gráficos de colapso de curvas...")

# Exponentes determinados en los pasos anteriores
exp_gamma_nu = slope_chi
exp_inv_nu   = -slope_w_chi
exp_alpha_nu = slope_cv
Tc_used      = Tc_extrapolada

# Panel para susceptibilidad: chi / L^(gamma/nu) vs (T - Tc) * L^(1/nu)
p_col_chi = plot(
    xlabel = "(T - Tc) L^{1/ν}",
    ylabel = "χ / L^{γ/ν}",
    title = @sprintf("Colapso χ (γ/ν = %.2f, 1/ν = %.2f, Tc = %.3f)", exp_gamma_nu, exp_inv_nu, Tc_used),
    legend = :topright
)

# Panel para calor específico: Cv / L^(alpha/nu) vs (T - Tc) * L^(1/nu)
p_col_cv = plot(
    xlabel = "(T - Tc) L^{1/ν}",
    ylabel = "Cv / L^{α/ν}",
    title = @sprintf("Colapso Cv (α/ν = %.2f, 1/ν = %.2f)", exp_alpha_nu, exp_inv_nu),
    legend = false
)

for (i, L) in enumerate(L_vec)
    df = datos_por_L[L]
    
    # Variable de escala adimensional en el eje x
    x_scaled = (df.T .- Tc_used) .* (Float64(L)^exp_inv_nu)
    
    # Magnitudes reescaladas en el eje y
    y_scaled_chi = df.susceptibilidad ./ (Float64(L)^exp_gamma_nu)
    y_scaled_cv  = df.calor_especifico ./ (Float64(L)^exp_alpha_nu)
    
    # Graficar con marcadores y línea suave
    plot!(p_col_chi, x_scaled, y_scaled_chi, 
          lw = 1.5, marker = :circle, markersize = 3, 
          color = paleta[i], label = "L = $L")
          
    plot!(p_col_cv, x_scaled, y_scaled_cv, 
          lw = 1.5, marker = :square, markersize = 3, 
          color = paleta[i], label = "L = $L")
end

# Línea vertical indicando el punto crítico x = 0
vline!(p_col_chi, [0.0], linestyle = :dash, color = :black, lw = 1, label = "")
vline!(p_col_cv,  [0.0], linestyle = :dash, color = :black, lw = 1, label = "")

# Centrar la visualización alrededor del régimen crítico si hay colas largas
# (Ajustable según el rango de temperaturas simulado)
# xlims!(p_col_chi, (-5.0, 5.0))
# xlims!(p_col_cv,  (-5.0, 5.0))

fig_collapse = plot(p_col_chi, p_col_cv, layout = (1, 2), size = (1000, 450), margin = 6mm, dpi = 200)
ruta_fig_col = joinpath(out_dir, "fss_data_collapse.png")
savefig(fig_collapse, ruta_fig_col)
println("Gráficos de colapso guardados en: $ruta_fig_col")