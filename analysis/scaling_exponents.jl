# analysis/scaling_exponents.jl
#
# Items (b), (e) y (g) de la Guia 4:
#   (b) Punto de inflexion de <|m|>(T), extrapolacion 1/L -> 0 y decaimiento 1/sqrt(N) a alta T
#   (e) Exponentes criticos por FSS en Tc y leyes de escala
#   (g) Crecimiento de max(Cv): ley de potencia vs divergencia logaritmica
#
# Requiere haber corrido antes critical_analysis.jl (fss_peaks_summary.csv)
# y binder_crossing.jl (binder_tc_estimate.csv).

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
    s = ArgParseSettings(description = "Inflexion de <|m|>, exponentes criticos y calor especifico (Ising 2D)")

    @add_arg_table! s begin
        "--data_dir"
            help = "Directorio donde residen los CSVs de mediciones (ej. equilibrium_observables_L*.csv)"
            arg_type = String
            default = joinpath(@__DIR__, "..", "run", "data")
        "--out_dir"
            help = "Directorio de destino (y de lectura de fss_peaks_summary.csv y binder_tc_estimate.csv)"
            arg_type = String
            default = joinpath(@__DIR__, "output")
        "--window_a"
            help = "Semiancho de las ventanas de ajuste local: a / L"
            arg_type = Float64
            default = 2.0
        "--degree"
            help = "Grado de los polinomios locales"
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
Ajuste polinomial local por cuadrados minimos (ponderado si se dan errores
`sigma`) en la ventana |T - T0| <= half_width. El polinomio esta en la
variable x = T - T0 para mejorar el condicionamiento.
"""
function local_poly(T::Vector{Float64}, Y::Vector{Float64}, T0::Float64, half_width::Float64, degree::Int;
                    sigma::Union{Nothing, Vector{Float64}} = nothing)
    mask = abs.(T .- T0) .<= half_width
    x = T[mask] .- T0
    y = Y[mask]
    n = length(x)
    n < degree + 3 && error("solo $n puntos en la ventana ±$(round(half_width, digits = 4)) alrededor de T = $(round(T0, digits = 4))")

    s = sigma === nothing ? ones(n) : sigma[mask]
    Aw = hcat([x .^ k for k in 0:degree]...) ./ s
    p = Polynomial(Aw \ (y ./ s))

    chi2_red = sum(((y .- p.(x)) ./ s) .^ 2) / (n - degree - 1)
    # Sin errores: el ruido se estima de los residuos. Con errores: se infla si chi2_red > 1
    noise_scale = sigma === nothing ? sqrt(chi2_red) : max(1.0, sqrt(chi2_red))

    return (p = p, T0 = T0, x = x, y = y, s = s, Aw = Aw, chi2_red = chi2_red, noise_scale = noise_scale)
end

"""
Error de una cantidad derivada f(polinomio) por bootstrap parametrico:
y* = ajuste + N(0, sigma) y se reajusta.
"""
function bootstrap_err(f, aj, n_boot::Int, rng)
    vals = Float64[]
    for _ in 1:n_boot
        y_star = aj.p.(aj.x) .+ aj.s .* aj.noise_scale .* randn(rng, length(aj.x))
        v = f(Polynomial(aj.Aw \ (y_star ./ aj.s)))
        isnan(v) || push!(vals, v)
    end
    return length(vals) > 1 ? std(vals) : NaN
end

"""
Punto de inflexion (raiz de p'' con maxima |p'|) dentro del rango de ajuste.
Devuelve x (relativo a T0) o NaN.
"""
function inflection_x(p::Polynomial, x_lo::Float64, x_hi::Float64)
    d1 = derivative(p)
    cands = [real(r) for r in roots(derivative(d1)) if abs(imag(r)) < 1e-10 && x_lo <= real(r) <= x_hi]
    isempty(cands) && return NaN
    return cands[argmax(abs.(d1.(cands)))]
end

"""
Regresion lineal por cuadrados minimos, ponderada si se dan errores `sigma`.
Sin errores, las incertezas se estiman de los residuos; con errores, se
inflan por sqrt(chi2_red) cuando este es mayor que 1.
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
Ajuste de ley de potencia y = A L^k en escala log-log (errores relativos si se dan).
"""
power_fit(L::Vector{Float64}, y::Vector{Float64}; sigma = nothing) =
    linear_fit(log.(L), log.(y); sigma = sigma === nothing ? nothing : sigma ./ y)

"""
Interpolacion lineal de (T, Y) en T_eval (T ordenado).
"""
function interp_linear(T::Vector{Float64}, Y::Vector{Float64}, T_eval::Float64)
    i = searchsortedlast(T, T_eval)
    i == length(T) && return Y[end]
    i == 0 && return Y[1]
    θ = (T_eval - T[i]) / (T[i + 1] - T[i])
    return (1 - θ) * Y[i] + θ * Y[i + 1]
end

# -------------------------------------------------------------------------
# Lectura de datos
# -------------------------------------------------------------------------
args = parse_commandline()

data_dir = abspath(args["data_dir"])
out_dir  = abspath(args["out_dir"])
window_a = args["window_a"]
degree   = args["degree"]
n_boot   = args["n_boot"]
mkpath(out_dir)

rng = Xoshiro(2027)

Tc_exacto = 2.269185
Kc = log(1 + sqrt(2)) / 2
B_cv_exacto = (8 / π) * Kc^2  # Cv_max ≈ B ln L + cte (Ferdinand & Fisher 1969)
exactos = Dict("beta" => 1 / 8, "gamma" => 7 / 4, "nu" => 1.0, "alpha" => 0.0, "eta" => 1 / 4, "delta" => 15.0)

println("="^75)
println(" INFLEXION, EXPONENTES CRITICOS Y CALOR ESPECIFICO - ISING 2D")
println("="^75)
println("Fecha y hora       : $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("Directorio datos   : $data_dir")
println("Ventanas locales   : ±$window_a / L, polinomio grado $degree")
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
L_f = Float64.(L_vec)
N_f = L_f .^ 2
println("Tamanos de red encontrados: $L_vec")

ruta_picos = joinpath(out_dir, "fss_peaks_summary.csv")
ruta_binder = joinpath(out_dir, "binder_tc_estimate.csv")
isfile(ruta_picos) || error("Falta $ruta_picos: correr primero critical_analysis.jl")
isfile(ruta_binder) || error("Falta $ruta_binder: correr primero binder_crossing.jl")

df_picos = sort(CSV.read(ruta_picos, DataFrame), :L)
df_picos.L == L_vec || error("Los tamanos de fss_peaks_summary.csv no coinciden con los datos: re-correr critical_analysis.jl")

df_binder = CSV.read(ruta_binder, DataFrame)
Tc = df_binder.Tc[1]
Tc_err = sqrt(df_binder.Tc_err_stat[1]^2 + df_binder.Tc_err_sist[1]^2)
@printf("Tc (Binder)        : %.5f +/- %.5f\n", Tc, Tc_err)

# =========================================================================
# (b) PUNTO DE INFLEXION DE <|m|>(T)
# =========================================================================
"""
Busca el punto de inflexion de <|m|>(T) para un L: arranca en la maxima
pendiente discreta y recentra la ventana sobre la inflexion del ajuste.
"""
function find_inflection(df::DataFrame, L::Int, a::Float64, deg::Int)
    T, m, σ = Float64.(df.T), Float64.(df.mag_abs), Float64.(df.mag_err)

    # Semilla: maxima pendiente por diferencias centradas
    dm = [-(m[i + 1] - m[i - 1]) / (T[i + 1] - T[i - 1]) for i in 2:length(T) - 1]
    T0 = T[argmax(dm) + 1]

    aj = nothing
    for _ in 1:5
        aj = local_poly(T, m, T0, a / L, deg; sigma = σ)
        x_inf = inflection_x(aj.p, minimum(aj.x), maximum(aj.x))
        isnan(x_inf) && error("L=$L: sin punto de inflexion en la ventana")
        abs(x_inf) < 1e-6 && break
        T0 += x_inf
    end
    x_inf = inflection_x(aj.p, minimum(aj.x), maximum(aj.x))
    return aj, T0 + x_inf
end

function inflection_table(a::Float64, deg::Int, nb::Int)
    filas = DataFrame(L = Int[], T_infl = Float64[], T_infl_err = Float64[], slope_infl = Float64[], chi2_red = Float64[])
    ajustes = Dict{Int, Any}()
    for L in L_vec
        aj, T_inf = find_inflection(datos_por_L[L], L, a, deg)
        err = bootstrap_err(q -> aj.T0 + inflection_x(q, minimum(aj.x), maximum(aj.x)), aj, nb, rng)
        slope = -derivative(aj.p)(T_inf - aj.T0)
        push!(filas, (L, T_inf, err, slope, aj.chi2_red))
        ajustes[L] = aj
    end
    return filas, ajustes
end

df_infl, ajustes_infl = inflection_table(window_a, degree, n_boot)

# Extrapolacion T_infl(L) = Tc + A L^(-1/nu), con nu = 1  ->  lineal en 1/L
fit_infl = linear_fit(1.0 ./ L_f, df_infl.T_infl; sigma = df_infl.T_infl_err)
sel32 = L_vec .> minimum(L_vec)
fit_infl_32 = linear_fit(1.0 ./ L_f[sel32], df_infl.T_infl[sel32]; sigma = df_infl.T_infl_err[sel32])

println("\n(b) Punto de inflexion de <|m|>:")
for r in eachrow(df_infl)
    @printf("  L = %-4d : T_infl = %.5f +/- %.5f  |d<|m|>/dT| = %6.2f  (chi2_red ajuste = %.2f)\n",
            r.L, r.T_infl, r.T_infl_err, r.slope_infl, r.chi2_red)
end
@printf("  Extrapolado (todos los L)  : Tc = %.5f +/- %.5f (chi2_red = %.2f)\n",
        fit_infl.intercept, fit_infl.intercept_err, fit_infl.chi2_red)
@printf("  Extrapolado (L >= %d)      : Tc = %.5f +/- %.5f (chi2_red = %.2f)\n",
        L_vec[2], fit_infl_32.intercept, fit_infl_32.intercept_err, fit_infl_32.chi2_red)

# Sistematicos de la extrapolacion
Tc_infl_sist_vals = Float64[]
for a in (1.5, 2.0, 3.0), deg in (3, 4)
    try
        filas, _ = inflection_table(a, deg, 300)
        f = linear_fit(1.0 ./ L_f, filas.T_infl; sigma = filas.T_infl_err)
        push!(Tc_infl_sist_vals, f.intercept)
        @printf("    sistematico a = %.1f, grado %d : Tc = %.5f +/- %.5f\n", a, deg, f.intercept, f.intercept_err)
    catch e
        @printf("    sistematico a = %.1f, grado %d : omitido (%s)\n", a, deg, sprint(showerror, e))
    end
end
Tc_infl_sist = (maximum(Tc_infl_sist_vals) - minimum(Tc_infl_sist_vals)) / 2

# -------------------------------------------------------------------------
# (b) Saturacion de alta temperatura: <|m|> ~ N^(-1/2)
# -------------------------------------------------------------------------
T_alta = [3.0, 3.5, 4.0]
df_alta = DataFrame(T = Float64[], L = Int[], N = Int[], mag_abs = Float64[], mag_err = Float64[], ratio_gauss = Float64[])
for Ta in T_alta, L in L_vec
    df = datos_por_L[L]
    T = Float64.(df.T)
    m = interp_linear(T, Float64.(df.mag_abs), Ta)
    e = interp_linear(T, Float64.(df.mag_err), Ta)
    m2 = interp_linear(T, Float64.(df.m2), Ta)
    # Para m gaussiana de media nula: <|m|> = sqrt(2 <m^2> / pi)
    push!(df_alta, (Ta, L, L^2, m, e, m / sqrt(2 * m2 / π)))
end

println("\n(b) Saturacion a alta T, <|m|> ~ N^k (esperado k = -1/2):")
fits_alta = Dict{Float64, Any}()
for Ta in T_alta
    sub = df_alta[df_alta.T .== Ta, :]
    f = power_fit(Float64.(sub.N), sub.mag_abs; sigma = sub.mag_err)
    fits_alta[Ta] = f
    @printf("  T = %.1f : k = %.4f +/- %.4f (chi2_red = %.2f) | <|m|>/sqrt(2<m²>/π) = %.3f ... %.3f\n",
            Ta, f.slope, f.slope_err, f.chi2_red, minimum(sub.ratio_gauss), maximum(sub.ratio_gauss))
end
CSV.write(joinpath(out_dir, "inflection_points.csv"), df_infl)
CSV.write(joinpath(out_dir, "high_T_magnetization.csv"), df_alta)

# =========================================================================
# (e) OBSERVABLES EN Tc Y EXPONENTES CRITICOS
# =========================================================================
df_tc = DataFrame(L = Int[], mag_Tc = Float64[], mag_Tc_err = Float64[], mag_Tc_lo = Float64[], mag_Tc_hi = Float64[],
                  dUdT_Tc = Float64[], dUdT_Tc_err = Float64[], chi_Tc = Float64[], chi_Tc_err = Float64[])
for L in L_vec
    df = datos_por_L[L]
    T = Float64.(df.T)
    w = window_a / L

    aj_m = local_poly(T, Float64.(df.mag_abs), Tc, w, degree; sigma = Float64.(df.mag_err))
    m_c = aj_m.p(0.0)
    m_c_err = bootstrap_err(q -> q(0.0), aj_m, n_boot, rng)

    aj_U = local_poly(T, Float64.(df.binder_cumulant), Tc, w, degree)
    dU = -derivative(aj_U.p)(0.0)
    dU_err = bootstrap_err(q -> -derivative(q)(0.0), aj_U, n_boot, rng)

    aj_chi = local_poly(T, Float64.(df.susceptibilidad), Tc, w, degree)
    chi_c = aj_chi.p(0.0)
    chi_c_err = bootstrap_err(q -> q(0.0), aj_chi, n_boot, rng)

    push!(df_tc, (L, m_c, m_c_err, aj_m.p(-Tc_err), aj_m.p(Tc_err), dU, dU_err, chi_c, chi_c_err))
end
CSV.write(joinpath(out_dir, "observables_at_Tc.csv"), df_tc)

# --- beta / nu: <|m|>(Tc) ~ L^(-beta/nu)
f_beta = power_fit(L_f, df_tc.mag_Tc; sigma = df_tc.mag_Tc_err)
f_beta_32 = power_fit(L_f[sel32], df_tc.mag_Tc[sel32]; sigma = df_tc.mag_Tc_err[sel32])
# Sensibilidad a Tc: se repite el ajuste evaluando en Tc -/+ sigma_Tc
beta_nu_Tc_sist = abs(power_fit(L_f, df_tc.mag_Tc_hi).slope - power_fit(L_f, df_tc.mag_Tc_lo).slope) / 2
beta_nu = -f_beta.slope
beta_nu_err = sqrt(f_beta.slope_err^2 + beta_nu_Tc_sist^2)

# --- 1/nu, tres estimaciones independientes
f_nu_U = power_fit(L_f, df_tc.dUdT_Tc; sigma = df_tc.dUdT_Tc_err)          # |dU/dT|(Tc) ~ L^(1/nu)
f_nu_shift = power_fit(L_f, df_picos.T_star_chi .- Tc)                       # T*(L) - Tc ~ L^(-1/nu)
f_nu_fwhm = power_fit(L_f, Float64.(df_picos.FWHM_chi))                      # FWHM(chi) ~ L^(-1/nu)

inv_nu = f_nu_U.slope
inv_nu_err = f_nu_U.slope_err
nu = 1 / inv_nu
nu_err = inv_nu_err / inv_nu^2

# --- gamma / nu: chi_max ~ L^(gamma/nu) (principal) y chi(Tc) ~ L^(gamma/nu)
f_gamma = power_fit(L_f, Float64.(df_picos.chi_max))
f_gamma_Tc = power_fit(L_f, df_tc.chi_Tc; sigma = df_tc.chi_Tc_err)
gamma_nu = f_gamma.slope
gamma_nu_err = f_gamma.slope_err

println("\n(e) Exponentes por escalamiento de tamano finito:")
@printf("  beta/nu  <|m|>(Tc) ~ L^-b/n        : %.4f +/- %.4f (ajuste %.4f, Tc %.4f; chi2_red = %.2f)  [L>=%d: %.4f +/- %.4f]  exacto 0.125\n",
        beta_nu, beta_nu_err, f_beta.slope_err, beta_nu_Tc_sist, f_beta.chi2_red, L_vec[2], -f_beta_32.slope, f_beta_32.slope_err)
@printf("  1/nu     |dU/dT|(Tc) ~ L^(1/nu)    : %.4f +/- %.4f (chi2_red = %.2f)  exacto 1\n", f_nu_U.slope, f_nu_U.slope_err, f_nu_U.chi2_red)
@printf("  1/nu     T*(L) - Tc ~ L^(-1/nu)    : %.4f +/- %.4f\n", -f_nu_shift.slope, f_nu_shift.slope_err)
@printf("  1/nu     FWHM(chi) ~ L^(-1/nu)     : %.4f +/- %.4f\n", -f_nu_fwhm.slope, f_nu_fwhm.slope_err)
@printf("  gamma/nu chi_max ~ L^(g/n)         : %.4f +/- %.4f  exacto 1.75\n", f_gamma.slope, f_gamma.slope_err)
@printf("  gamma/nu chi(Tc) ~ L^(g/n)         : %.4f +/- %.4f (chi2_red = %.2f)\n", f_gamma_Tc.slope, f_gamma_Tc.slope_err, f_gamma_Tc.chi2_red)

# --- Leyes de escala (d = 2), con propagacion de errores por Monte Carlo
d = 2
n_mc = 200_000
s_bn = beta_nu .+ beta_nu_err .* randn(rng, n_mc)
s_gn = gamma_nu .+ gamma_nu_err .* randn(rng, n_mc)
s_in = inv_nu .+ inv_nu_err .* randn(rng, n_mc)
s_nu = 1 ./ s_in

derivados = [
    ("nu",    "1 / (1/nu)",            s_nu),
    ("beta",  "nu * (beta/nu)",        s_nu .* s_bn),
    ("gamma", "nu * (gamma/nu)",       s_nu .* s_gn),
    ("alpha", "2 - d nu (hiperesc.)",  2 .- d .* s_nu),
    ("eta",   "2 - gamma/nu (Fisher)", 2 .- s_gn),
    ("delta", "1 + gamma/beta (Widom)", 1 .+ s_gn ./ s_bn),
]

df_exp = DataFrame(exponente = String[], relacion = String[], valor = Float64[], error = Float64[],
                   exacto = Float64[], desvio_sigma = Float64[])
println("\n(e) Exponentes derivados por leyes de escala (d = 2):")
for (nombre, rel, muestras) in derivados
    v, e = median(muestras), std(muestras)
    ex = exactos[nombre]
    push!(df_exp, (nombre, rel, v, e, ex, (v - ex) / e))
    @printf("  %-6s = %8.4f +/- %.4f   [%-23s]  exacto %7.4f  (%+.1f sigma)\n", nombre, v, e, rel, ex, (v - ex) / e)
end
hiperesc = 2 * beta_nu + gamma_nu
hiperesc_err = sqrt((2 * beta_nu_err)^2 + gamma_nu_err^2)
@printf("  Chequeo hiperescala 2 beta/nu + gamma/nu = d : %.4f +/- %.4f (d = 2)\n", hiperesc, hiperesc_err)
CSV.write(joinpath(out_dir, "exponents_summary.csv"), df_exp)

# =========================================================================
# (g) CALOR ESPECIFICO: max(Cv) vs L
# =========================================================================
cv_max = Float64.(df_picos.cv_max)
f_cv_log = linear_fit(log.(L_f), cv_max)
f_cv_pow = power_fit(L_f, cv_max)
rms_log = sqrt(mean((cv_max .- (f_cv_log.intercept .+ f_cv_log.slope .* log.(L_f))) .^ 2))
rms_pow = sqrt(mean((cv_max .- exp(f_cv_pow.intercept) .* L_f .^ f_cv_pow.slope) .^ 2))
alpha_nu_eff = [log(cv_max[i + 1] / cv_max[i]) / log(L_f[i + 1] / L_f[i]) for i in 1:length(L_vec) - 1]
L_eff = [sqrt(L_f[i] * L_f[i + 1]) for i in 1:length(L_vec) - 1]

println("\n(g) Calor especifico maximo:")
@printf("  Cv_max = A + B ln L : B = %.4f +/- %.4f (exacto %.4f), A = %.4f, rms = %.4f\n",
        f_cv_log.slope, f_cv_log.slope_err, B_cv_exacto, f_cv_log.intercept, rms_log)
@printf("  Cv_max ~ L^(a/n)    : alpha/nu = %.4f +/- %.4f, rms = %.4f\n", f_cv_pow.slope, f_cv_pow.slope_err, rms_pow)
println("  alpha/nu efectivo entre tamanos consecutivos:")
for i in eachindex(alpha_nu_eff)
    @printf("    L = %3d -> %3d : %.4f\n", L_vec[i], L_vec[i + 1], alpha_nu_eff[i])
end

# =========================================================================
# Gráficos
# =========================================================================
println("\nGenerando figuras...")
paleta = palette(:thermal, length(L_vec) + 1)

# --- FIGURA (b)
p_dm = plot(xlabel = "T", ylabel = "-d<|m|>/dT", title = "Pendiente de <|m|> (ajustes locales)", legend = :topright)
for (i, L) in enumerate(L_vec)
    aj = ajustes_infl[L]
    x_fino = range(minimum(aj.x), maximum(aj.x), length = 200)
    plot!(p_dm, aj.T0 .+ x_fino, -derivative(aj.p).(x_fino), color = paleta[i], lw = 2, label = "L = $L")
    scatter!(p_dm, [df_infl.T_infl[i]], [df_infl.slope_infl[i]], color = paleta[i], markersize = 5, label = "")
end
vline!(p_dm, [Tc_exacto], linestyle = :dash, color = :black, lw = 1, label = "Tc Onsager")

x_ext = [0.0; 1.0 ./ L_f]
p_ext = scatter(1.0 ./ L_f, df_infl.T_infl, yerror = df_infl.T_infl_err, color = :blue, markersize = 5,
                label = "T_infl(L)", xlabel = "1/L", ylabel = "T de inflexion", title = "Extrapolacion 1/L -> 0",
                legend = :topleft)
plot!(p_ext, x_ext, fit_infl.intercept .+ fit_infl.slope .* x_ext, color = :black, lw = 1.5,
      label = @sprintf("Ajuste: Tc = %.4f(%d)", fit_infl.intercept, round(Int, 1e4 * fit_infl.intercept_err)))
hline!(p_ext, [Tc_exacto], linestyle = :dash, color = :gray, label = "Tc Onsager")

p_alta = plot(xlabel = "N = L²", ylabel = "<|m|>", title = "Saturacion a alta T ~ N^(-1/2)",
              xscale = :log10, yscale = :log10, legend = :bottomleft)
for (k, Ta) in enumerate(T_alta)
    sub = df_alta[df_alta.T .== Ta, :]
    f = fits_alta[Ta]
    scatter!(p_alta, sub.N, sub.mag_abs, yerror = sub.mag_err, markersize = 5, color = k,
             label = @sprintf("T = %.1f: pendiente %.3f", Ta, f.slope))
    plot!(p_alta, sub.N, exp(f.intercept) .* Float64.(sub.N) .^ f.slope, color = k, lw = 1.5, label = "")
end

fig_b = plot(p_dm, p_ext, p_alta, layout = @layout([a; b c]), size = (1000, 850), margin = 6mm, dpi = 200)
savefig(fig_b, joinpath(out_dir, "magnetization_inflection.png"))

# --- FIGURA (e)
function loglog_panel(y, yerr, f, titulo, ylab, etiqueta)
    p = scatter(L_f, y, yerror = yerr, xscale = :log2, yscale = :log10, markersize = 5, color = :red,
                xticks = (L_f, string.(L_vec)), label = "Datos", xlabel = "L", ylabel = ylab, title = titulo,
                legend = :best)
    plot!(p, L_f, exp(f.intercept) .* L_f .^ f.slope, color = :black, lw = 1.5, label = etiqueta)
    return p
end

p_e1 = loglog_panel(df_tc.mag_Tc, df_tc.mag_Tc_err, f_beta, "<|m|>(Tc) ~ L^(-β/ν)", "<|m|>(Tc)",
                    @sprintf("β/ν = %.4f", beta_nu))
p_e2 = loglog_panel(df_tc.dUdT_Tc, df_tc.dUdT_Tc_err, f_nu_U, "|dU/dT|(Tc) ~ L^(1/ν)", "|dU/dT|(Tc)",
                    @sprintf("1/ν = %.4f", f_nu_U.slope))
p_e3 = loglog_panel(df_picos.T_star_chi .- Tc, nothing, f_nu_shift, "T*(L) - Tc ~ L^(-1/ν)", "T*(L) - Tc",
                    @sprintf("1/ν = %.4f", -f_nu_shift.slope))
p_e4 = loglog_panel(Float64.(df_picos.chi_max), nothing, f_gamma, "χ_max ~ L^(γ/ν)", "χ",
                    @sprintf("γ/ν = %.4f", f_gamma.slope))
scatter!(p_e4, L_f, df_tc.chi_Tc, color = :orange, markersize = 4, label = @sprintf("χ(Tc): γ/ν = %.3f", f_gamma_Tc.slope))

fig_e = plot(p_e1, p_e2, p_e3, p_e4, layout = (2, 2), size = (1000, 800), margin = 6mm, dpi = 200)
savefig(fig_e, joinpath(out_dir, "critical_exponents_fss.png"))

# --- FIGURA (g)
lnL_fino = range(log(L_f[1]) - 0.2, log(L_f[end]) + 0.2, length = 100)
p_g1 = scatter(log.(L_f), cv_max, color = :purple, markersize = 6, label = "max(Cv)",
               xlabel = "ln L", ylabel = "Cv_max", title = "Divergencia logaritmica", legend = :topleft)
plot!(p_g1, lnL_fino, f_cv_log.intercept .+ f_cv_log.slope .* lnL_fino, color = :black, lw = 1.5,
      label = @sprintf("A + B ln L, B = %.3f", f_cv_log.slope))
plot!(p_g1, lnL_fino, mean(cv_max .- B_cv_exacto .* log.(L_f)) .+ B_cv_exacto .* lnL_fino,
      color = :gray, linestyle = :dash, lw = 1.5, label = @sprintf("Pendiente exacta %.4f", B_cv_exacto))

p_g2 = scatter(L_eff, alpha_nu_eff, color = :purple, markersize = 6, xscale = :log2,
               xticks = (L_eff, [@sprintf("%d-%d", L_vec[i], L_vec[i + 1]) for i in eachindex(L_eff)]),
               label = "Local", xlabel = "Par de tamaños", ylabel = "α/ν efectivo",
               title = "Exponente efectivo de Cv_max", legend = :topright, ylims = (0, maximum(alpha_nu_eff) * 1.2))
hline!(p_g2, [f_cv_pow.slope], color = :black, lw = 1.5, label = @sprintf("Ajuste global %.3f", f_cv_pow.slope))
# Para Cv = A + B ln L, el exponente efectivo local es B / Cv
plot!(p_g2, L_eff, B_cv_exacto ./ (f_cv_log.intercept .+ B_cv_exacto .* log.(L_eff)),
      color = :gray, linestyle = :dash, lw = 1.5, label = "B / Cv (log puro)")

fig_g = plot(p_g1, p_g2, layout = (1, 2), size = (1000, 430), margin = 6mm, dpi = 200)
savefig(fig_g, joinpath(out_dir, "specific_heat_scaling.png"))

# =========================================================================
# Reporte
# =========================================================================
ruta_reporte = joinpath(out_dir, "scaling_exponents_report.txt")
open(ruta_reporte, "w") do io
    println(io, "="^78)
    println(io, " REPORTE: INFLEXION DE <|m|>, EXPONENTES CRITICOS Y CALOR ESPECIFICO (ISING 2D)")
    println(io, "="^78)
    println(io, "Fecha y hora del analisis : $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io, "Directorio de datos       : $data_dir")
    println(io, "Tamanos L procesados      : $(join(L_vec, ", "))")
    println(io, "Ventanas locales          : ±$window_a / L, polinomio grado $degree")
    @printf(io, "Tc usado (Binder)         : %.5f +/- %.5f\n", Tc, Tc_err)
    println(io, "-"^78)
    println(io, "(b) PUNTO DE INFLEXION DE <|m|>(T)")
    for r in eachrow(df_infl)
        @printf(io, "  L = %3d : T_infl = %.5f +/- %.5f\n", r.L, r.T_infl, r.T_infl_err)
    end
    @printf(io, "  Tc extrapolado (1/L -> 0)  : %.5f +/- %.5f (estad.) +/- %.5f (sist.)  [Onsager 2.269185]\n",
            fit_infl.intercept, fit_infl.intercept_err, Tc_infl_sist)
    @printf(io, "  Tc extrapolado (L >= %d)    : %.5f +/- %.5f\n", L_vec[2], fit_infl_32.intercept, fit_infl_32.intercept_err)
    println(io, "  Saturacion a alta T, <|m|> ~ N^k (esperado k = -0.5):")
    for Ta in T_alta
        @printf(io, "    T = %.1f : k = %.4f +/- %.4f\n", Ta, fits_alta[Ta].slope, fits_alta[Ta].slope_err)
    end
    println(io, "-"^78)
    println(io, "(e) EXPONENTES POR FSS")
    @printf(io, "  beta/nu  (<|m|>(Tc))        : %.4f +/- %.4f   (exacto 0.125)\n", beta_nu, beta_nu_err)
    @printf(io, "  1/nu     (|dU/dT|(Tc))      : %.4f +/- %.4f   (exacto 1)\n", f_nu_U.slope, f_nu_U.slope_err)
    @printf(io, "  1/nu     (T*(L) - Tc)       : %.4f +/- %.4f\n", -f_nu_shift.slope, f_nu_shift.slope_err)
    @printf(io, "  1/nu     (FWHM chi)         : %.4f +/- %.4f\n", -f_nu_fwhm.slope, f_nu_fwhm.slope_err)
    @printf(io, "  gamma/nu (chi_max)          : %.4f +/- %.4f   (exacto 1.75)\n", f_gamma.slope, f_gamma.slope_err)
    @printf(io, "  gamma/nu (chi(Tc))          : %.4f +/- %.4f\n", f_gamma_Tc.slope, f_gamma_Tc.slope_err)
    println(io, "  Exponentes derivados (beta/nu, gamma/nu de chi_max, 1/nu de dU/dT):")
    for r in eachrow(df_exp)
        @printf(io, "    %-6s = %8.4f +/- %.4f  [%-23s]  exacto %7.4f (%+.1f sigma)\n",
                r.exponente, r.valor, r.error, r.relacion, r.exacto, r.desvio_sigma)
    end
    @printf(io, "  Hiperescala 2 beta/nu + gamma/nu = %.4f +/- %.4f (d = 2)\n", hiperesc, hiperesc_err)
    println(io, "-"^78)
    println(io, "(g) CALOR ESPECIFICO MAXIMO")
    @printf(io, "  Cv_max = A + B ln L : B = %.4f +/- %.4f (exacto %.4f), rms = %.4f\n",
            f_cv_log.slope, f_cv_log.slope_err, B_cv_exacto, rms_log)
    @printf(io, "  Cv_max ~ L^(a/n)    : alpha/nu = %.4f +/- %.4f, rms = %.4f\n", f_cv_pow.slope, f_cv_pow.slope_err, rms_pow)
    println(io, "  alpha/nu efectivo   : " * join([@sprintf("%d-%d: %.3f", L_vec[i], L_vec[i + 1], alpha_nu_eff[i]) for i in eachindex(alpha_nu_eff)], ", "))
    println(io, "="^78)
end
println("Reporte guardado en: $ruta_reporte")
