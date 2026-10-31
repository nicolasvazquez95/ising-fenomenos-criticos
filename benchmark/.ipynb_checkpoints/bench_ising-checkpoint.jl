# benchmark/bench_ising.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using BenchmarkTools
using Random
using Printf
using Plots, Measures
using DataFrames
using CSV

# 1. Cargar el código fuente
include(joinpath(@__DIR__, "..", "src", "ising.jl"))

println("="^72)
println(" BENCHMARK ISING 2D: Escalamiento por tamaño de red (L x L)")
println("="^72)

# 2. Parámetros físicos comunes
J = 1.0
h = 0.1
T = 2.269               # Tc ≈ 2.269
niters = 100            # Pasos MC por evaluación
tamanos_L = [16, 32, 64, 128, 256, 512, 1024]

# Inicializamos el DataFrame para almacenar los resultados
df_resultados = DataFrame(
    L = Int[],
    N = Int[],
    tiempo_ms = Float64[],
    mflips_s = Float64[],
    memoria_kib = Float64[],
    allocs = Int[]
)

rng_setup = Xoshiro(1234)

# 3. Bucle de benchmarking
for L in tamanos_L
    m, n = L, L
    N = m * n
    total_flips = N * niters
    
    s_init = rand(rng_setup, Int8[-1, 1], m, n)
    
    b = @benchmark ising2d!(s, $J, $h, $T, $niters) setup=(s = copy($s_init)) seconds=2
    
    t_med_s = median(b).time * 1e-9
    t_med_ms = t_med_s * 1e3
    mflips = (total_flips / t_med_s) / 1e6
    mem_kib = b.memory / 1024
    allocs = b.allocs
    
    push!(df_resultados, (L, N, t_med_ms, mflips, mem_kib, allocs))
    println("Completado: L = $L ($N espines)")
end

# 4. Tabla formateada en terminal
println("\n" * "="^72)
@printf("%-6s | %-8s | %-12s | %-12s | %-10s | %-8s\n", 
        "L", "N (sitios)", "Tiempo (ms)", "MFlips/s", "Memoria", "Allocs")
println("-"^72)

for row in eachrow(df_resultados)
    @printf("%-6d | %-10d | %-12.3f | %-12.2f | %-7.2f KiB | %-8d\n", 
            row.L, row.N, row.tiempo_ms, row.mflips_s, row.memoria_kib, row.allocs)
end
println("="^72)

# 5. Exportar el DataFrame a CSV
ruta_csv = joinpath(@__DIR__, "benchmark_ising_results.csv")
CSV.write(ruta_csv, df_resultados)
println("Resultados exportados a DataFrame y guardados en: $ruta_csv")

# 6. Generación y guardado de gráficos
println("Generando figuras...")

p1 = plot(df_resultados.N, df_resultados.tiempo_ms,
    marker = :circle,
    linewidth = 2,
    xlabel = "Número de espines N (L²)",
    ylabel = "Tiempo mediano (ms)",
    title = "Escalamiento temporal",
    xscale = :log10,
    yscale = :log10,
    label = "ising2d! (niters=$niters)",
    legend = :topleft
)

p2 = plot(df_resultados.L, df_resultados.mflips_s,
    marker = :square,
    linewidth = 2,
    xlabel = "Tamaño lateral L",
    ylabel = "MFlips / s",
    title = "Rendimiento vs tamaño",
    xscale = :log2,
    label = "MFlips/s",
    legend = :bottomleft
)

plot_final = plot(p1, p2, layout = (1, 2), size = (900, 420), dpi=300, margin=5mm)
ruta_figura = joinpath(@__DIR__, "benchmark_ising_results.png")
savefig(plot_final, ruta_figura)
println("Gráfico guardado con éxito en: $ruta_figura")