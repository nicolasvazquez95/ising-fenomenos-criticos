# run/append_moments.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using DataFrames
using CSV
using Printf
using Dates
using ArgParse

function parse_commandline()
    s = ArgParseSettings(description = "Calcula e inserta m2 y m4 en los equilibrium_observables_L*.csv existentes")

    @add_arg_table! s begin
        "--data_dir"
            help = "Directorio donde estan guardados los CSVs"
            arg_type = String
            default = joinpath(@__DIR__, "data")
    end

    return parse_args(s)
end

args = parse_commandline()
data_dir = abspath(args["data_dir"])

println("="^75)
println(" ACTUALIZACION DE OBSERVABLES: CALCULO DE <m^2> Y <m^4> (ISING 2D)")
println("="^75)
println("Directorio de datos: $data_dir\n")

patron = Regex("^equilibrium_observables_L(\\d+)\\.csv\$")
archivos = filter(f -> occursin(patron, f), readdir(data_dir))

if isempty(archivos)
    error("No se encontraron archivos 'equilibrium_observables_L<L>.csv' en $data_dir")
end

for arch in archivos
    m = match(patron, arch)
    L = parse(Int, m.captures[1])
    N = L * L
    ruta_csv = joinpath(data_dir, arch)

    df = CSV.read(ruta_csv, DataFrame)

    # Verificar si ya tienen las columnas
    if in("m2", names(df)) && in("m4", names(df))
        println("-> [L=$L] El archivo ya cuenta con 'm2' y 'm4'. Omitiendo.")
        continue
    end

    # Despeje analítico exacto fila por fila
    # chi = (N / T) * (m2 - mag_abs^2) => m2 = (T * chi / N) + mag_abs^2
    m2_calc = (df.T .* df.susceptibilidad) ./ N .+ (df.mag_abs .^ 2)
    
    # U_L = 1 - m4 / (3 * m2^2) => m4 = 3 * m2^2 * (1 - U_L)
    m4_calc = 3.0 .* (m2_calc .^ 2) .* (1.0 .- df.binder_cumulant)

    # Insertamos las columnas de forma ordenada
    df.m2 = m2_calc
    df.m4 = m4_calc

    # Reordenamiento estético de columnas
    cols_orden = [
        :T, :mag_abs, :mag_err, :m2, :m4,
        :ene_per_spin, :ene_err,
        :susceptibilidad, :calor_especifico, :binder_cumulant,
        :sample_interval, :n_therm
    ]
    # Filtramos por las que existan en el DataFrame original
    cols_finales = [c for c in cols_orden if in(string(c), names(df))]
    df_actualizado = df[:, cols_finales]

    # Respaldo de seguridad previo y escritura atómica
    ruta_backup = joinpath(data_dir, "equilibrium_observables_L$(L)_sin_m2m4_backup.csv")
    cp(ruta_csv, ruta_backup; force = true)

    ruta_tmp = joinpath(data_dir, "equilibrium_observables_L$(L).tmp")
    CSV.write(ruta_tmp, df_actualizado)
    mv(ruta_tmp, ruta_csv; force = true)

    println("✓ [L=$L] Columnas m2 y m4 agregadas con exito.")
    println("       Respaldo guardado en: $(basename(ruta_backup))")
end

println("\n" * "="^75)
println("Todos los archivos fueron actualizados y respaldados correctamente.")
println("="^75)