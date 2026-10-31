# analysis/field_scaling.jl
#
# Item (f) de la Guia 4:
#   - Curvas m(B/J) para temperaturas por encima, por debajo y en la Tc estimada
#   - Exponente delta a partir de m ~ B^(1/delta) en Tc
#   - Colapso m/|t|^beta vs B/|t|^Delta (Delta = beta delta) con exponentes
#     exactos y estimados
#
# Requiere run/field_magnetization.jl (field_magnetization_L*.csv) y, para los
# exponentes estimados, scaling_exponents.jl (exponents_summary.csv) y
# binder_crossing.jl (binder_tc_estimate.csv).

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
    s = ArgParseSettings(description = "Magnetizacion vs campo: exponente delta y colapso de escala (Ising 2D)")

    @add_arg_table! s begin
        "--data_dir"
            help = "Directorio donde residen los field_magnetization_L*.csv"
            arg_type = String
            default = joinpath(@__DIR__, "..", "run", "data")
        "--out_dir"
            help = "Directorio de destino (y de lectura de exponents_summary.csv y binder_tc_estimate.csv)"
            arg_type = String
            default = joinpath(@__DIR__, "output")
        "--L"
            help = "Tamano de red principal para el analisis"
            arg_type = Int
            default = 200
        "--B_fit_max"
            help = "Campo maximo del ajuste m ~ B^(1/delta) en Tc (por defecto: la mayor ventana con chi2_red <= 1)"
            arg_type = Float64
            default = 0.022
        "--t_scaling_max"
            help = "|t| maximo considerado dentro del regimen de escala (metrica de colapso)"
            arg_type = Float64
            default = 0.05
        "--B_scaling_max"
            help = "Campo maximo considerado dentro del regimen de escala (metrica de colapso)"
            arg_type = Float64
            default = 0.1
    end

    return parse_args(s)
end

# -------------------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------------------
"""
Regresion lineal por cuadrados minimos, ponderada si se dan errores `sigma`.
Con errores, las incertezas se inflan por sqrt(chi2_red) cuando este es mayor que 1.
"""
function linear_fit(x::Vector{Float64}, y::Vector{Float64}; sigma::Union{Nothing, Vector{Float64}} = nothing)
    n = length(x)
    s = sigma === nothing ? ones(n) : sigma
    w = 1.0 ./ s .^ 2

    S, Sx, Sy = sum(w), sum(w .* x), sum(w .* y)
    Sxx, Sxy = sum(w .* x .^ 2), sum(w .* x .* y)
    Δ = S * Sxx - Sx^2

    slope = (S * Sxy - Sx * Sy) / Δ
    intercept = (Sxx * Sy - Sx * Sxy) / Δ

    chi2_red = sum(w .* (y .- intercept .- slope .* x) .^ 2) / max(1, n - 2)
    scale = sigma === nothing ? sqrt(chi2_red) : max(1.0, sqrt(chi2_red))

    return (slope = slope, intercept = intercept,
            slope_err = sqrt(S / Δ) * scale, intercept_err = sqrt(Sxx / Δ) * scale,
            chi2_red = chi2_red)
end

"""
Ajuste de ln m = ln A + (1/delta) ln B con errores relativos de m.
Devuelve el ajuste y delta con su error.
"""
function delta_fit(B::Vector{Float64}, m::Vector{Float64}, m_err::Vector{Float64})
    f = linear_fit(log.(B), log.(m); sigma = m_err ./ m)
    delta = 1 / f.slope
    return f, delta, f.slope_err / f.slope^2
end

"""
Dispersion del colapso: rms de la diferencia en ln(y) entre cada curva y la
interpolacion lineal (en escala log-log) de las demas curvas de la misma rama,
dentro de su rango comun de x.
"""
function collapse_dispersion(curvas::Vector{Tuple{Vector{Float64}, Vector{Float64}}})
    difs = Float64[]
    for (i, (lx_i, ly_i)) in enumerate(curvas), (j, (lx_j, ly_j)) in enumerate(curvas)
        i == j && continue
        length(lx_j) < 2 && continue
        for (x, y) in zip(lx_i, ly_i)
            (lx_j[1] <= x <= lx_j[end]) || continue
            k = clamp(searchsortedlast(lx_j, x), 1, length(lx_j) - 1)
            θ = (x - lx_j[k]) / (lx_j[k + 1] - lx_j[k])
            push!(difs, y - ((1 - θ) * ly_j[k] + θ * ly_j[k + 1]))
        end
    end
    return isempty(difs) ? NaN : sqrt(mean(difs .^ 2)), length(difs)
end

# -------------------------------------------------------------------------
# Lectura de datos
# -------------------------------------------------------------------------
args = parse_commandline()

data_dir = abspath(args["data_dir"])
out_dir  = abspath(args["out_dir"])
L_main   = args["L"]
B_fit_max = args["B_fit_max"]
t_sc_max = args["t_scaling_max"]
B_sc_max = args["B_scaling_max"]
mkpath(out_dir)

beta_ex, delta_ex = 1 / 8, 15.0

println("="^75)
println(" MAGNETIZACION VS CAMPO: EXPONENTE DELTA Y COLAPSO DE ESCALA - ISING 2D")
println("="^75)
println("Fecha y hora       : $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("Directorio datos   : $data_dir")

patron = Regex("^field_magnetization_L(\\d+)\\.csv\$")
archivos = filter(f -> occursin(patron, f), readdir(data_dir))
isempty(archivos) && error("No se encontraron archivos 'field_magnetization_L<L>.csv' en $data_dir")

datos_por_L = Dict{Int, DataFrame}()
for arch in archivos
    L_val = parse(Int, match(patron, arch).captures[1])
    df = CSV.read(joinpath(data_dir, arch), DataFrame)
    sort!(df, [:t, :B])
    datos_por_L[L_val] = df
end
haskey(datos_por_L, L_main) || error("No hay datos para L = $L_main")
df = datos_por_L[L_main]
Tc_used = df.Tc_used[1]
t_vals = sort(unique(df.t))
println("Tamanos con datos  : $(sort(collect(keys(datos_por_L)))) (principal: L = $L_main, $(nrow(df)) puntos)")
@printf("Tc de referencia   : %.6f\n", Tc_used)
println("Puntos ok = false  : $(count(.!df.ok)) (incluidos: su error ya contempla la correlacion residual)")

# Exponentes estimados en (e) y error de Tc
ruta_exp = joinpath(out_dir, "exponents_summary.csv")
ruta_binder = joinpath(out_dir, "binder_tc_estimate.csv")
isfile(ruta_exp) || error("Falta $ruta_exp: correr primero scaling_exponents.jl")
isfile(ruta_binder) || error("Falta $ruta_binder: correr primero binder_crossing.jl")
df_exp = CSV.read(ruta_exp, DataFrame)
beta_est = df_exp.valor[df_exp.exponente .== "beta"][1]
beta_est_err = df_exp.error[df_exp.exponente .== "beta"][1]
delta_widom = df_exp.valor[df_exp.exponente .== "delta"][1]
delta_widom_err = df_exp.error[df_exp.exponente .== "delta"][1]
df_binder = CSV.read(ruta_binder, DataFrame)
Tc_err = sqrt(df_binder.Tc_err_stat[1]^2 + df_binder.Tc_err_sist[1]^2)

# =========================================================================
# 1. Exponente delta en Tc: m ~ B^(1/delta)
# =========================================================================
d0 = df[df.t .== 0.0, :]
B0, m0, e0 = Float64.(d0.B), Float64.(d0.mag), Float64.(d0.mag_err)

sel = B0 .<= B_fit_max
f_main, delta_est, delta_est_err_fit = delta_fit(B0[sel], m0[sel], e0[sel])

# Sensibilidad a Tc: ln m se interpola linealmente en t entre t = 0 y t = ±0.01
eps_t = Tc_err / Tc_used
function m_shifted(signo)
    dv = df[df.t .== signo * 0.01, :]
    @assert dv.B ≈ d0.B "Las grillas de campo de t = 0 y t = ±0.01 no coinciden"
    return exp.(log.(m0) .+ (eps_t / 0.01) .* (log.(Float64.(dv.mag)) .- log.(m0)))
end
_, delta_hi, _ = delta_fit(B0[sel], m_shifted(+1)[sel], e0[sel])
_, delta_lo, _ = delta_fit(B0[sel], m_shifted(-1)[sel], e0[sel])
delta_Tc_sist = abs(delta_hi - delta_lo) / 2

# Sistematicos: ventana de ajuste
df_ventanas = DataFrame(B_min = Float64[], B_max = Float64[], n = Int[], delta = Float64[],
                        delta_err = Float64[], chi2_red = Float64[])
for Bmin in B0[1:3], Bmax in B0[B0 .>= 0.009 .&& B0 .<= 0.25]
    s = (B0 .>= Bmin * (1 - 1e-9)) .& (B0 .<= Bmax * (1 + 1e-9))
    count(s) < 4 && continue
    f, dlt, dlt_e = delta_fit(B0[s], m0[s], e0[s])
    push!(df_ventanas, (Bmin, Bmax, count(s), dlt, dlt_e, f.chi2_red))
end
CSV.write(joinpath(out_dir, "field_delta_fits.csv"), df_ventanas)
# Dispersion entre ventanas con ajuste aceptable (chi2_red <= 1): a campos mayores
# aparecen correcciones de escala y la ley de potencia pura deja de describir los datos
sub_v = df_ventanas[df_ventanas.chi2_red .<= 1.0, :]
delta_win_sist = (maximum(sub_v.delta) - minimum(sub_v.delta)) / 2

delta_err_tot = sqrt(delta_est_err_fit^2 + delta_Tc_sist^2 + delta_win_sist^2)

# Exponente efectivo local 1/delta_eff entre campos consecutivos
inv_delta_eff = [log(m0[i + 1] / m0[i]) / log(B0[i + 1] / B0[i]) for i in 1:length(B0) - 1]
inv_delta_eff_err = [sqrt((e0[i + 1] / m0[i + 1])^2 + (e0[i] / m0[i])^2) / log(B0[i + 1] / B0[i]) for i in 1:length(B0) - 1]
B_eff = [sqrt(B0[i] * B0[i + 1]) for i in 1:length(B0) - 1]

println("\n(f) Exponente delta en Tc (m ~ B^(1/delta), L = $L_main):")
@printf("  Ajuste B in [%.0e, %.2g] (%d puntos): 1/delta = %.5f +/- %.5f, chi2_red = %.2f\n",
        minimum(B0[sel]), maximum(B0[sel]), count(sel), f_main.slope, f_main.slope_err, f_main.chi2_red)
@printf("  delta = %.3f +/- %.3f (ajuste) +/- %.3f (Tc) +/- %.3f (ventana) -> +/- %.3f total   [exacto 15]\n",
        delta_est, delta_est_err_fit, delta_Tc_sist, delta_win_sist, delta_err_tot)
@printf("  delta por Widom (item e): %.2f +/- %.2f\n", delta_widom, delta_widom_err)
println("  Todas las ventanas (las de chi2_red <= 1 definen el sistematico):")
for r in eachrow(df_ventanas)
    @printf("    B in [%.1e, %.3f] (%2d pts): delta = %6.3f +/- %.3f  chi2_red = %.2f\n",
            r.B_min, r.B_max, r.n, r.delta, r.delta_err, r.chi2_red)
end

# =========================================================================
# 2. Colapso m/|t|^beta vs B/|t|^Delta
# =========================================================================
juegos = [
    ("Exactos", beta_ex, delta_ex),
    ("Estimados", beta_est, delta_est),
]

function scaled_curves(β, δ; t_max = Inf, B_max = Inf)
    Δ = β * δ
    ramas = Dict(:baja => Tuple{Vector{Float64}, Vector{Float64}}[], :alta => Tuple{Vector{Float64}, Vector{Float64}}[])
    for t in t_vals
        (t == 0.0 || abs(t) > t_max + 1e-12) && continue
        d = df[(df.t .== t) .& (df.B .<= B_max * (1 + 1e-9)) .& (df.mag .> 0), :]
        nrow(d) < 2 && continue
        lx = log.(Float64.(d.B) ./ abs(t)^Δ)
        ly = log.(Float64.(d.mag) ./ abs(t)^β)
        push!(ramas[t < 0 ? :baja : :alta], (lx, ly))
    end
    return ramas
end

println("\n(f) Calidad del colapso: rms de Δln(m/|t|^β) entre curvas de la misma rama")
df_colapso = DataFrame(juego = String[], beta = Float64[], delta = Float64[], Delta = Float64[],
                       region = String[], rms_baja = Float64[], rms_alta = Float64[])
for (nombre, β, δ) in juegos, (region, tm, bm) in (("todos los puntos", Inf, Inf),
                                                   (@sprintf("|t| <= %.2f, B <= %.2f", t_sc_max, B_sc_max), t_sc_max, B_sc_max))
    ramas = scaled_curves(β, δ; t_max = tm, B_max = bm)
    rb, _ = collapse_dispersion(ramas[:baja])
    ra, _ = collapse_dispersion(ramas[:alta])
    push!(df_colapso, (nombre, β, δ, β * δ, region, rb, ra))
    @printf("  %-9s (β = %.4f, δ = %6.3f, Δ = %.4f) | %-24s | T < Tc: %.4f   T > Tc: %.4f\n",
            nombre, β, δ, β * δ, region, rb, ra)
end
CSV.write(joinpath(out_dir, "field_collapse_quality.csv"), df_colapso)

# =========================================================================
# 3. Comparacion entre tamaños (efectos de tamaño finito)
# =========================================================================
otros_L = sort([L for L in keys(datos_por_L) if L != L_main])
df_fss = DataFrame(L = Int[], n_comun = Int[], rms_pull = Float64[], max_abs_pull = Float64[],
                   t_max_pull = Float64[], B_max_pull = Float64[])
for L in otros_L
    dL = datos_por_L[L]
    comun = innerjoin(select(df, :t, :B, :mag => :m1, :mag_err => :e1),
                      select(dL, :t, :B, :mag => :m2, :mag_err => :e2), on = [:t, :B])
    nrow(comun) == 0 && continue
    pull = (comun.m1 .- comun.m2) ./ sqrt.(comun.e1 .^ 2 .+ comun.e2 .^ 2)
    k = argmax(abs.(pull))
    push!(df_fss, (L, nrow(comun), sqrt(mean(pull .^ 2)), abs(pull[k]), comun.t[k], comun.B[k]))
end
if nrow(df_fss) > 0
    println("\n(f) Comparacion con otros tamaños (pull = Δm / σ, se espera rms ≈ 1 sin efectos de tamaño finito):")
    for r in eachrow(df_fss)
        @printf("  L = %d vs %d : %3d puntos comunes, rms pull = %.2f, max |pull| = %.1f (t = %+.2f, B = %.1e)\n",
                L_main, r.L, r.n_comun, r.rms_pull, r.max_abs_pull, r.t_max_pull, r.B_max_pull)
    end
end

# =========================================================================
# Gráficos
# =========================================================================
println("\nGenerando figuras...")
paleta_t = cgrad(:coolwarm, length(t_vals), categorical = true)
color_t(t) = t == 0.0 ? :black : paleta_t[findfirst(==(t), t_vals)]
etiqueta_t(t) = t == 0.0 ? "t = 0 (Tc)" : @sprintf("t = %+.2f", t)

# --- FIGURA 1: curvas m(B)
p_log = plot(xlabel = "B / J", ylabel = "m", title = "Magnetizacion vs campo (L = $L_main)",
             xscale = :log10, legend = :outerright, ylims = (0, 1.02))
p_lin = plot(xlabel = "B / J", ylabel = "m", title = "m(B) (B < 0 por simetria m(-B) = -m(B))",
             legend = false, xlims = (-1, 1))
for t in t_vals
    d = df[df.t .== t, :]
    plot!(p_log, d.B, d.mag, yerror = d.mag_err, color = color_t(t), lw = t == 0 ? 2.5 : 1.5,
          marker = :circle, markersize = 2.5, markerstrokewidth = 0, label = etiqueta_t(t))
    dd = d[d.B .<= 1.0, :]
    plot!(p_lin, [-reverse(dd.B); dd.B], [-reverse(dd.mag); dd.mag], color = color_t(t), lw = t == 0 ? 2.5 : 1.5)
end
hline!(p_lin, [0.0], color = :gray, lw = 0.5)
vline!(p_lin, [0.0], color = :gray, lw = 0.5)
fig1 = plot(p_log, p_lin, layout = (1, 2), size = (1200, 480), margin = 6mm, dpi = 200)
savefig(fig1, joinpath(out_dir, "field_magnetization_curves.png"))

# --- FIGURA 2: delta en Tc
p_d1 = scatter(B0, m0, yerror = e0, xscale = :log10, yscale = :log10, color = :black, markersize = 4,
               label = "m(B) en Tc", xlabel = "B / J", ylabel = "m", title = "m ~ B^(1/δ) en Tc",
               legend = :topleft)
B_fino = exp10.(range(log10(minimum(B0)), log10(B_fit_max), length = 50))
plot!(p_d1, B_fino, exp(f_main.intercept) .* B_fino .^ f_main.slope, color = :red, lw = 2,
      label = @sprintf("Ajuste B ≤ %.2g: δ = %.2f(%d)", B_fit_max, delta_est, round(Int, 100 * delta_err_tot)))
plot!(p_d1, B_fino, exp(f_main.intercept) .* B_fino .^ (1 / delta_ex), color = :gray, lw = 1.5,
      linestyle = :dash, label = "Pendiente exacta 1/15")

p_d2 = scatter(B_eff, inv_delta_eff, yerror = inv_delta_eff_err, xscale = :log10, color = :black, markersize = 4,
               label = "Local", xlabel = "B / J", ylabel = "d ln m / d ln B", title = "Exponente efectivo 1/δ_eff",
               legend = :bottomleft)
hline!(p_d2, [1 / delta_ex], color = :gray, linestyle = :dash, lw = 1.5, label = "1/15 exacto")
hline!(p_d2, [f_main.slope], color = :red, lw = 1.5, label = "Ajuste")
vline!(p_d2, [B_fit_max], color = :red, linestyle = :dot, lw = 1, label = "Limite del ajuste")

fig2 = plot(p_d1, p_d2, layout = (1, 2), size = (1100, 450), margin = 6mm, dpi = 200)
savefig(fig2, joinpath(out_dir, "field_delta.png"))

# --- FIGURA 3: colapso con exponentes exactos y estimados
function panel_colapso(nombre, β, δ)
    Δ = β * δ
    p = plot(xscale = :log10, yscale = :log10, xlabel = "B / |t|^Δ", ylabel = "m / |t|^β",
             title = @sprintf("%s: β = %.4f, δ = %.2f, Δ = %.4f", nombre, β, δ, Δ), legend = :bottomright)
    for t in t_vals
        t == 0.0 && continue
        d = df[(df.t .== t) .& (df.mag .> 0), :]
        x = d.B ./ abs(t)^Δ
        y = d.mag ./ abs(t)^β
        plot!(p, x, y, color = color_t(t), lw = 1.5, marker = t < 0 ? :circle : :utriangle, markersize = 3,
              markerstrokewidth = 0, label = etiqueta_t(t))
    end
    # Comportamiento asintotico en x -> ∞: m/|t|^β ~ A x^(1/δ), con A de m(Tc) ~ A B^(1/δ)
    x_as = exp10.(range(0, 4, length = 50))
    A = exp(f_main.intercept)
    plot!(p, x_as, A .* x_as .^ (1 / δ), color = :black, linestyle = :dash, lw = 1.5, label = "A x^(1/δ) (Tc)")
    return p
end

fig3 = plot(panel_colapso("Exactos", beta_ex, delta_ex), panel_colapso("Estimados", beta_est, delta_est),
            layout = (1, 2), size = (1300, 560), margin = 7mm, dpi = 200)
savefig(fig3, joinpath(out_dir, "field_scaling_collapse.png"))
println("Gráficos guardados en: $out_dir")

# =========================================================================
# Reporte
# =========================================================================
ruta_reporte = joinpath(out_dir, "field_scaling_report.txt")
open(ruta_reporte, "w") do io
    println(io, "="^78)
    println(io, " REPORTE: MAGNETIZACION VS CAMPO, EXPONENTE DELTA Y COLAPSO (ISING 2D)")
    println(io, "="^78)
    println(io, "Fecha y hora del analisis : $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io, "Datos                     : field_magnetization_L$(L_main).csv ($(nrow(df)) puntos)")
    @printf(io, "Tc de referencia          : %.6f +/- %.5f (Binder)\n", Tc_used, Tc_err)
    println(io, "Temperaturas (t = T/Tc-1) : $(join(t_vals, ", "))")
    println(io, "-"^78)
    println(io, "EXPONENTE DELTA (m ~ B^(1/delta) en Tc)")
    @printf(io, "  Ventana principal       : B in [%.0e, %.2g], %d puntos, chi2_red = %.2f\n",
            minimum(B0[sel]), maximum(B0[sel]), count(sel), f_main.chi2_red)
    @printf(io, "  delta                   : %.3f +/- %.3f (ajuste) +/- %.3f (Tc) +/- %.3f (ventana)\n",
            delta_est, delta_est_err_fit, delta_Tc_sist, delta_win_sist)
    @printf(io, "  delta (error total)     : %.3f +/- %.3f   [exacto 15]\n", delta_est, delta_err_tot)
    @printf(io, "  delta por Widom (item e): %.2f +/- %.2f\n", delta_widom, delta_widom_err)
    println(io, "-"^78)
    println(io, "COLAPSO m/|t|^beta vs B/|t|^Delta (rms de Δln y entre curvas de una rama)")
    for r in eachrow(df_colapso)
        @printf(io, "  %-9s beta = %.4f, delta = %6.3f, Delta = %.4f | %-24s | T<Tc: %.4f  T>Tc: %.4f\n",
                r.juego, r.beta, r.delta, r.Delta, r.region, r.rms_baja, r.rms_alta)
    end
    if nrow(df_fss) > 0
        println(io, "-"^78)
        println(io, "COMPARACION ENTRE TAMAÑOS (pull = Δm / σ)")
        for r in eachrow(df_fss)
            @printf(io, "  L = %d vs %d : %d puntos, rms pull = %.2f, max |pull| = %.1f (t = %+.2f, B = %.1e)\n",
                    L_main, r.L, r.n_comun, r.rms_pull, r.max_abs_pull, r.t_max_pull, r.B_max_pull)
        end
    end
    println(io, "="^78)
end
println("Reporte guardado en: $ruta_reporte")
