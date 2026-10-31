# Guía interna del proyecto (no es el informe)

Mapa de dónde está cada resultado de la Guía 4, qué archivo y qué columnas usar
para cada gráfico, y notas de metodología para tener en cuenta al escribir.
Todas las rutas son relativas a `~/ising-fenomenos-criticos/`.

---

## 0. Cómo está organizado y cómo regenerar todo

### Pipeline

| Etapa | Script | Produce |
|---|---|---|
| Núcleo del Monte Carlo | `src/ising.jl` (`ising2d!`) | — |
| Observables (m, E) | `run/runising.jl` | — |
| 1. Estudio de autocorrelación | `run/decorrelation_study.jl` (+ `resume_decorrelation_study.jl`) | `run/data/decorrelation_L*_nT96.csv`, `autocorrelation_curves_*`, `timeseries_*` |
| 2. Producción en equilibrio (B = 0) | `run/production_run.jl` | `run/data/equilibrium_observables_L*.csv` |
| 3. Extensión de colas a T ∈ [1, 4] | `run/extend_tails.jl` | mismo CSV (respaldo `_orig_backup.csv`) |
| 4. Agregar ⟨m²⟩, ⟨m⁴⟩ | `run/append_moments.jl` | mismo CSV (respaldo `_sin_m2m4_backup.csv`) |
| 5. Corridas con campo (ítem f) | `run/field_magnetization.jl` | `run/data/field_magnetization_L{128,200}.csv` |
| Extra: histéresis | `run/hysteresis_loop.jl` | `run/data/hysteresis_L64.*` |
| Análisis FSS básico (picos de χ y Cv) | `analysis/critical_analysis.jl` | `fss_peaks_summary.csv`, ... |
| Cruce de Binder (ítem c) | `analysis/binder_crossing.jl` | `binder_*.csv`, `binder_tc_estimate.csv` |
| Inflexión, exponentes, Cv (b, e, g) | `analysis/scaling_exponents.jl` | `inflection_points.csv`, `exponents_summary.csv`, ... |
| Comparación con Onsager (ítem a) | `analysis/observables_vs_exact.jl` | `exact_comparison.csv`, ... |
| Campo: δ y colapso (ítem f) | `analysis/field_scaling.jl` | `field_*.csv` |

Todas las salidas de análisis van a `analysis/output/`.

### Orden para regenerar los análisis (minutos)

Hay dependencias: `critical_analysis` y `binder_crossing` no dependen de otros
análisis, `scaling_exponents` lee
`fss_peaks_summary.csv` y `binder_tc_estimate.csv`, `field_scaling` lee
`exponents_summary.csv` y `binder_tc_estimate.csv`.

```
cd ~/ising-fenomenos-criticos
J=~/.juliaup/bin/julia
$J analysis/critical_analysis.jl
$J analysis/binder_crossing.jl
$J analysis/scaling_exponents.jl
$J analysis/observables_vs_exact.jl
$J analysis/field_scaling.jl
```

**Usar siempre `~/.juliaup/bin/julia`** (1.13). El `julia` que aparece primero en
el PATH (`~/.local/bin/julia`, 1.12) no coincide con el `Manifest.toml` y falla
al cargar Plots (error con `Zstd_jll`).

### Archivos que NO hay que usar

- `analysis/data/`: copias viejas de L = 16 y 32, anteriores a la extensión de
  colas y sin `m2`/`m4`. Los datos vigentes están en `run/data/`.
- `run/data/equilibrium_observables_L*.png`: figuras generadas por
  `production_run.jl` en el momento de la corrida; muestran solo el rango
  original de T (sin colas). Sirven como control, no para el informe.
- `run/data/*_orig_backup.csv` y `*_sin_m2m4_backup.csv`: respaldos.
- `.ipynb_checkpoints/` en cualquier carpeta: copias de Jupyter.
- Las series temporales crudas del estudio de decorrelación de L = 128 y 200
  (~75 GB) ya no están en el proyecto: se movieron a `~/ising_archivos_pesados/`
  (ver sección 6).

---

## 1. Datos principales: `run/data/equilibrium_observables_L{16,32,64,128,200}.csv`

Un archivo por L, 128 filas cada uno, T ∈ [1, 4] con **grilla no uniforme**:
densa cerca de Tc (rango de producción) y más gruesa en las colas.

| Columna | Qué es |
|---|---|
| `T` | temperatura (J = k_B = 1) |
| `mag_abs`, `mag_err` | ⟨\|m\|⟩ y su error σ/√n |
| `m2`, `m4` | ⟨m²⟩, ⟨m⁴⟩ |
| `ene_per_spin`, `ene_err` | ⟨e⟩ = ⟨H⟩/N y su error |
| `susceptibilidad` | χ' = (N/T)(⟨m²⟩ − ⟨\|m\|⟩²) |
| `calor_especifico` | C = (1/(N T²))(⟨H²⟩ − ⟨H⟩²) |
| `binder_cumulant` | U_L = 1 − ⟨m⁴⟩/(3⟨m²⟩²) |
| `sample_interval`, `n_therm` | MCS entre muestras y de termalización usados en ese punto |

Rango de producción (zona densa) por L:
L = 16: [1.5, 3.5]; L = 32: [1.75, 3.0]; L = 64: [2.1, 2.5]; L = 128: [2.2, 2.4];
L = 200: [2.23, 2.33]. Fuera de eso son las colas de `extend_tails.jl`.

---

## 2. Resultados por ítem

### Problema 1 — χ y C como fluctuaciones

**Pendiente** (derivación analítica, no hay nada en el código). Las fórmulas que
usa el código coinciden con las de la guía: χ = βN(⟨m²⟩ − ⟨m⟩²) y
C = (β²/N)(⟨H²⟩ − ⟨H⟩²). Ver la nota sobre χ' en la sección 3.4.

### (a) Curvas de ⟨|m|⟩, χ, C para todos los L + solución exacta + ⟨m²⟩, ⟨m⁴⟩

- **Datos**: los CSV de la sección 1 (columnas `mag_abs`, `susceptibilidad`,
  `calor_especifico`, `m2`, `m4`).
- **Solución exacta**: implementada en `analysis/observables_vs_exact.jl`
  (funciones `mag_exact`, `ene_exact`, `cv_exact`). Si graficás por tu cuenta,
  se pueden copiar de ahí: magnetización de Yang, energía y Cv de Onsager con
  integrales elípticas por AGM. Están verificadas internamente (Cv = du/dT a
  10⁻⁹, u(Tc) = −√2).
- **Figuras de referencia**:
  - `analysis/output/observables_vs_exact.png`: ⟨|m|⟩, ⟨e⟩, χ (log), Cv en
    rango completo, con exacto.
  - `analysis/output/observables_near_Tc.png`: zoom 2.1–2.5 de ⟨|m|⟩ y Cv, y
    ⟨m²⟩, ⟨m⁴⟩ (comparados con m², m⁴ exactos).
  - `analysis/output/all_observables_curves.png` (de `critical_analysis.jl`):
    versión anterior sin exacto; reemplazada por las dos de arriba.
- **Comparación cuantitativa con el exacto** (|T − Tc| > 0.25):
  `analysis/output/exact_comparison.csv` y `observables_vs_exact_report.txt`.
  Para L ≥ 32 el rms del pull (Δ/σ) de ⟨|m|⟩ y ⟨e⟩ da 0.84–1.03 → las
  diferencias son estadísticas. L = 16 en energía da 2.4 (efecto de tamaño
  finito real). Cv: desvío medio < 1 %, rms 2–4 %.
- **"Tests preliminares"** que pide el enunciado: el estudio de decorrelación
  (sección 3.2). Figuras: `run/data/decorrelation_L*_nT96.png` (ρ(t) y
  τ_int(T)). Tabla: `run/data/decorrelation_L*_nT96.csv` (columnas `T`,
  `tau_int_mag`, `tau_int_ene`). Picos de τ_int(|m|): ≈ 62, 284, 1220, ~5500,
  ~14000 MCS para L = 16…200 (→ ralentización crítica, z ≈ 2.1–2.2).

### (b) Punto de inflexión de ⟨|m|⟩ y saturación 1/√N

- **Datos**: `analysis/output/inflection_points.csv` (`L`, `T_infl`,
  `T_infl_err`, `slope_infl` = |d⟨|m|⟩/dT| en el punto) y
  `analysis/output/high_T_magnetization.csv` (`T` ∈ {3, 3.5, 4}, `L`, `N`,
  `mag_abs`, `mag_err`, `ratio_gauss`).
- **Figura**: `analysis/output/magnetization_inflection.png` (derivadas
  locales, extrapolación 1/L, log-log de ⟨|m|⟩ vs N).
- **Resultados** (`scaling_exponents_report.txt`):
  - T_infl: 2.3433, 2.3091, 2.2902, 2.2805, 2.2770 (L = 16…200).
  - Extrapolación lineal en 1/L: **Tc = 2.2712 ± 0.0014 (estad.) ± 0.0037 (sist.)**.
  - Alta T: ⟨|m|⟩ ∝ N^k con k = −0.498, −0.505, −0.498 (T = 3, 3.5, 4).
  - `ratio_gauss` = ⟨|m|⟩/√(2⟨m²⟩/π) ≈ 1 (±1 %): la distribución de m es
    gaussiana centrada en 0 → explica el 1/√N (N espines ~ independientes,
    teorema central del límite). Buen argumento para el informe.

### (c) Tc por cruce del cumulante de Binder

- **Datos**: `analysis/output/binder_crossings.csv` (todos los pares L1 < L2:
  `T_cross`, `U_cross` con errores), `binder_systematics.csv` (variación de
  ventana y grado), `binder_tc_estimate.csv` (resultado final).
  Para graficar U_L(T): columna `binder_cumulant` de los CSV de la sección 1.
- **Figura**: `analysis/output/binder_crossing.png` (zoom del cruce con ajustes,
  cruces de pares consecutivos, colapso de U_L con ν = 1).
- **Resultados**: **Tc = 2.26928 ± 0.00035 (estad.) ± 0.00016 (sist.)**, a 0.3σ
  de Onsager. U* = 0.6106 ± 0.0012 (referencia universal 0.61069).
- **Convención**: la guía define U_L = ⟨m⁴⟩/⟨m²⟩²; el código usa la de Binder
  1 − ⟨m⁴⟩/(3⟨m²⟩²). Son equivalentes (U_guía = 3(1 − U_código)); el cruce es el
  mismo. En la convención de la guía, U* = 1.168 ± 0.004 (ref. 1.1679). Si
  graficás con la de la guía, recalculá a partir de `m2` y `m4`.

### (d) Máximo de χ: T*(L) y γ/ν

- **Datos**: `analysis/output/fss_peaks_summary.csv` (`L`, `T_star_chi`,
  `chi_max`, `FWHM_chi`, y lo mismo para Cv).
- **Figura**: `analysis/output/fss_scaling_fits.png` (paneles χ_max vs L,
  FWHM vs L, T* vs 1/L, Cv_max vs ln L).
- **Resultados** (`critical_exponents_report.txt`):
  - **γ/ν = 1.7557 ± 0.0065** (exacto 1.75).
  - T*(L) → **Tc = 2.2701 ± 0.0011** (extrapolación lineal en 1/L, es decir
    asumiendo ν = 1).

### (e) Todos los exponentes por leyes de escala

- **Datos**: `analysis/output/exponents_summary.csv` (exponente, relación usada,
  valor, error, exacto, desvío en σ) y `observables_at_Tc.csv` (⟨|m|⟩, |dU/dT| y
  χ evaluados en el Tc de Binder, por L).
- **Figura**: `analysis/output/critical_exponents_fss.png` (log-log de
  ⟨|m|⟩(Tc), |dU/dT|(Tc), T* − Tc, χ_max).
- **Mediciones directas**:

  | Cantidad | Método | Valor | Exacto |
  |---|---|---|---|
  | β/ν | ⟨\|m\|⟩(Tc) ∼ L^(−β/ν) | 0.1257 ± 0.0034 | 0.125 |
  | 1/ν | \|dU/dT\|(Tc) ∼ L^(1/ν) | 0.977 ± 0.016 | 1 |
  | 1/ν | T*(L) − Tc ∼ L^(−1/ν) | 1.002 ± 0.018 | 1 |
  | 1/ν | FWHM(χ) ∼ L^(−1/ν) | 1.031 ± 0.012 | 1 |
  | γ/ν | χ_max | 1.7557 ± 0.0065 | 1.75 |
  | γ/ν | χ(Tc) | 1.7607 ± 0.0046 | 1.75 |

- **Derivados** (con β/ν, γ/ν de χ_max y 1/ν de dU/dT; errores por Monte Carlo):
  ν = 1.024(17), β = 0.129(4), γ = 1.80(3), α = 2 − dν = −0.05(3),
  η = 2 − γ/ν = 0.244(7), δ = 1 + γ/β = 14.97(39). Todos a < 1.6σ del exacto.
  Chequeo de hiperescala: 2β/ν + γ/ν = 2.007 ± 0.010 (d = 2).
- Elegí 1/ν de dU/dT como principal porque no depende de la grilla de T (el
  FWHM sí, ver 3.5). Si preferís otro, cambia ν y todo lo que depende de él.

### (f) m(B), δ y colapso m/|t|^β vs B/|t|^Δ

- **Datos**: `run/data/field_magnetization_L200.csv` (principal) y `_L128.csv`
  (control). Columnas útiles: `t`, `T`, `B`, `mag`, `mag_err`, `mag_abs`, `chi`,
  `ok`. Usar **`mag` (con signo)**, no `mag_abs`. Grilla: 11 temperaturas
  T = Tc(1 + t), t ∈ {0, ±0.01, ±0.02, ±0.05, ±0.1, ±0.2}, con Tc = 2.26928
  (Binder); 21 campos (19 log-espaciados en [10⁻³, 1] más B = 2 y 5).
- **Figuras**: `analysis/output/field_magnetization_curves.png` (m vs B en log y
  lineal; la rama B < 0 es reflejo por simetría, no está simulada),
  `field_delta.png` (ajuste en Tc y exponente efectivo local),
  `field_scaling_collapse.png` (colapso con exponentes exactos y estimados).
- **Resultados** (`field_scaling_report.txt`):
  - **δ = 15.05 ± 0.19** (L = 200; ajuste B ∈ [10⁻³, 0.022], χ²_red = 0.74).
    Error = ajuste 0.06 ⊕ Tc 0.14 ⊕ ventana 0.11. Con L = 128: 15.19 ± 0.29
    (`analysis/output/check_L128/`).
  - Tablas de todas las ventanas de ajuste: `field_delta_fits.csv`. δ sube
    hasta ~15.4–15.8 si se incluyen campos ≳ 0.05 (correcciones de escala,
    χ²_red crece). Por eso la ventana chica.
  - Colapso (rms de Δln y entre curvas, región |t| ≤ 0.05, B ≤ 0.1):
    exactos 0.008 / 0.011 (T < Tc / T > Tc); estimados (β = 0.1286, δ = 15.05,
    Δ = 1.936) 0.010 / 0.021. El estimado colapsa algo peor porque β (de e)
    está 0.9σ alto y Δ = βδ lo amplifica. `field_collapse_quality.csv`.
  - L = 128 vs 200: rms del pull = 0.93 en 231 puntos → sin efectos de
    tamaño finito en este rango de campos.
- **Puntos con `ok = false`** (8 en L = 200, 5 en L = 128): el τ de la corrida
  piloto se subestimó y las 3000 muestras quedaron algo correlacionadas
  (τ_muestras 1.06–1.52 en vez de ≈ 0.5). Están incluidos: su `mag_err` ya está
  corregido por σ√(2τ_muestras/n) y la termalización alcanzó igual (el piloto
  cubre ≥ 100 τ_piloto). Ver 3.7.
- En 8 puntos de t ≥ +0.05 y B ≲ 3·10⁻³, `mag` ≠ `mag_abs`: no es tunneling,
  es que m es chica (0.01–0.08) y sus fluctuaciones cruzan el cero. Físicamente
  correcto; la cantidad a usar con campo es ⟨m⟩.

### (g) Crecimiento de Cv_max con L

- **Datos**: `analysis/output/fss_peaks_summary.csv` (`L`, `cv_max`).
- **Figuras**: `analysis/output/specific_heat_scaling.png` (Cv_max vs ln L con
  ajuste y pendiente exacta; exponente efectivo α/ν entre pares de L) y el
  panel 4 de `fss_scaling_fits.png`.
- **Resultados**:
  - Ley de potencia (lo que pide la guía): **α/ν = 0.24 ± 0.01** (aparente).
  - Logarítmico: Cv_max = A + B ln L con **B = 0.518 ± 0.006** (exacto
    (8/π)K_c² = 0.4945, Ferdinand–Fisher). Residuo rms 4× menor que la potencia.
  - α/ν efectivo entre tamaños consecutivos: 0.31, 0.24, 0.20, 0.21 → decrece,
    como predice B/Cv para una divergencia log pura. Argumento: α = 0 y el α/ν
    aparente es un artefacto de ajustar potencia a un log.

### Extra (no pedido): ciclos de histéresis

`run/data/hysteresis_L64.png`, `hysteresis_L64.csv` (`T`, `B`, `m_baja`,
`m_sube`), `hysteresis_summary_L64.csv` (campo coercitivo, remanencia, área).
L = 64, barrido ±1.5 J a 10⁻³ J/MCS, promedio de 5 ciclos. Hay ciclo aun por
encima de Tc: es histéresis dinámica por velocidad de barrido finita, no
metaestabilidad (se achica con `--mcs_per_step` más grande).

---

## 3. Notas de metodología

### 3.1 Algoritmo

- **No es Metropolis**: `ising2d!` usa **baño térmico (heat bath / Glauber)**:
  se elige un sitio al azar y se lo fija en +1 con probabilidad
  1/(1 + e^(−2β(J k + B))), con k la suma de los 4 vecinos (tabla precalculada).
  La guía pide Metropolis. Las propiedades de equilibrio son las mismas (ambos
  cumplen balance detallado con la distribución de Boltzmann); lo que cambia es
  la dinámica, es decir los valores de τ (y z). Conviene aclararlo en el informe.
- **Selección aleatoria de sitios** (no barrido secuencial). 1 MCS = N intentos.
- Condiciones periódicas, J = k_B = 1, spins `Int8`.
- RNG Xoshiro. Semillas: decorrelación 42 + 1000·idx; producción
  2026 + 10000·idx; colas 4096 + 31L + 10000·idx; campo 2028 + 10000·iT + iB;
  histéresis 4242 + 100·i. Todo reproducible.
- Rendimiento (`benchmark/`): ≈ 185 Mflips/s por hilo hasta L = 256.

### 3.2 Estudio de autocorrelación (decorrelation_study.jl)

- Cold start (todo +1), series de |m| y E en cada MCS; se descartan los primeros
  min(5000, n/3) MCS.
- τ_int con **ventana autoconsistente de Sokal**: τ = 1/2 + Σ ρ(t), cortando en
  el primer t ≥ 5τ.
- Largos: 0.5 M MCS (L = 16) hasta 20 M MCS (L = 128, 200).
- **Advertencia**: para L = 128 y 200 la estimación de τ es ruidosa, sobre todo
  por debajo de Tc (por ejemplo L = 128, T = 2.234 da 6187 contra ~1000 en los
  vecinos; L = 200 salta entre 2845 y 12002 cerca de 2.25). Además, para L = 200
  el descarte de 5000 MCS es menor que τ (~14000), aunque con 20 M MCS de serie
  el efecto es chico. Estos τ solo se usaron para fijar el muestreo de
  producción, así que el ruido se traduce en sobremuestreo (más costo), no en
  sesgo.

### 3.3 Producción (production_run.jl) y colas (extend_tails.jl)

- τ(T) se **interpola linealmente** del CSV de decorrelación; fuera del rango
  medido se usa el valor del borde.
- **Termalización**: max(500, ⌈10τ⌉) MCS. **Separación entre muestras**:
  max(2, ⌈2.5τ⌉) MCS. Cold start.
- Muestras por T: 7500 (L = 16), 5000 (L = 32), 6000 (L = 64), 6500 (L = 128),
  6000 (L = 200).
- **Detalle**: para L = 32, `production_run.jl` tomó `decorrelation_L32_nT15.csv`
  (el estudio grueso de 15 puntos) en lugar de `_nT96`, porque elige el primer
  archivo que coincide con el patrón. Consecuencia menor: el τ usado es el
  del estudio grueso (pico 201 en vez de 284), o sea muestras separadas ~1.8τ
  en lugar de 2.5τ cerca de Tc. La comparación con el exacto para L = 32 da
  pull rms ≈ 1, así que no se nota en los resultados.
- **Colas**: `extend_tails.jl` agregó T fuera del rango de producción hasta
  [1, 4] usando como τ el valor del borde del rango (estimado desde
  `sample_interval` del borde / 2.5). Lejos de Tc el τ real es menor, así que es
  conservador. Por eso, para L = 200, todas las colas tienen
  `sample_interval` = 30005: muy sobremuestreado, pero correcto.
- Tiempo de producción L = 200: ~27 h con 64 hilos.

### 3.4 Definiciones de observables

- **χ' con ⟨|m|⟩**: χ' = (N/T)(⟨m²⟩ − ⟨|m|⟩²). En red finita a B = 0, ⟨m⟩ = 0 por
  simetría, así que usar ⟨m⟩ daría χ = N⟨m²⟩/T, que diverge con N en la fase
  ordenada. χ' es el estimador estándar de red finita (Binder–Landau). Por
  encima de Tc, χ' es menor que la χ "verdadera" N⟨m²⟩/T (factor 1 − 2/π para
  m gaussiana); el exponente γ/ν es el mismo. El pico de χ' está corrido hacia
  arriba de Tc más que el de N⟨m²⟩/T (de ahí T*(L) bastante por encima de Tc
  para L chico).
- **Cv** se calcula con la energía total (incluye el término de campo, que a
  B = 0 no aporta): C = Var(H)/(N T²) = (β²/N)(⟨H²⟩ − ⟨H⟩²). Coincide con la guía.
- **⟨m²⟩ y ⟨m⁴⟩** no se guardaron durante la producción; `append_moments.jl`
  los reconstruye **exactamente** despejándolos de χ' y U_L:
  ⟨m²⟩ = T χ'/N + ⟨|m|⟩², ⟨m⁴⟩ = 3⟨m²⟩²(1 − U_L). No es una aproximación: son las
  mismas medias muestrales (verificado a 10⁻¹⁶).

### 3.5 Errores y ajustes

- `mag_err`, `ene_err` = σ/√n, válidos porque las muestras están separadas 2.5τ.
  La comparación con el exacto (pull rms ≈ 1 para L ≥ 32) lo confirma.
- **χ, Cv y U_L no tienen error por punto** (no se guardaron las muestras crudas,
  así que no hay jackknife/bootstrap posible sin volver a correr). Las barras de
  error de todo lo que sale de χ_max, T*, FWHM, Cv_max vienen de los **residuos
  de los ajustes** (regresión lineal con error escalado por √χ²_red): son
  orientativas.
- Para cantidades derivadas de U_L y de ajustes locales se usó **bootstrap
  paramétrico** con la dispersión de los residuos (Binder, dU/dT, inflexión).
  Donde había `mag_err`, se usó como peso y como ruido del bootstrap.
- **Ventanas de ajuste locales**: |T − T0| ≤ a/L (a = 2 por defecto), es decir
  escalan como L^(−1/ν) con ν = 1, para cubrir un rango similar de la variable
  de escala en todos los tamaños. Sistemáticos estimados variando a ∈ {1.5, 2, 3}
  y el grado del polinomio.
- **Picos** (`critical_analysis.jl`): parábola local de ±3 puntos alrededor del
  máximo discreto. **FWHM**: cruce de media altura en los puntos de la grilla,
  **sin interpolar** → depende del espaciado. Por eso 1/ν del FWHM es el menos
  confiable.
- Error en el Tc extrapolado de T*(L): corregido el 2026-10-29 (antes se
  reportaba el error de la pendiente en lugar del de la ordenada; ±0.035 → ±0.0011).

### 3.6 Comparación de las estimaciones de Tc

| Método | Tc | Comentario |
|---|---|---|
| Cruce de Binder (c) | 2.26928 ± 0.00038 | la más precisa, sin extrapolación |
| T*(L) de χ' (d) | 2.2701 ± 0.0011 | extrapolación lineal en 1/L (ν = 1) |
| Inflexión de ⟨\|m\|⟩ (b) | 2.2712 ± 0.0040 | muy sensible a la ventana y al grado |
| Onsager | 2.269185 | |

Todas compatibles. El Tc de Binder es el que se usó como referencia en (e) y (f).

### 3.7 Corridas con campo (field_magnetization.jl)

- Mismo protocolo que producción (2.5τ, 10τ), pero **τ se estima punto a punto**
  con una corrida piloto (no había estudio de decorrelación con campo): cada MCS
  se mide m y el piloto se alarga ×4 hasta cubrir ≥ 100τ.
- Secuencia por punto: 500 MCS de burn-in → piloto → max(500, 10τ) de
  termalización → 3000 muestras separadas max(2, ⌈2.5τ⌉).
- Control: τ de la serie de 3000 muestras (`tau_samples`, ≈ 0.5 si son
  independientes; mediana 0.61). Error de la media σ√(2τ_muestras/n): coincide
  con σ/√n cuando son independientes y corrige cuando no. `ok = false` si
  τ_muestras ≥ 1 o si el piloto no convergió.
- Cold start alineado con B > 0 → no hay metaestabilidad por debajo de Tc.
- L = 200: B·L^(15/8) ≈ 21 en el campo mínimo ⇒ el campo fija el signo de m (no
  hay tunneling; en L = 32 sí lo había a B = 10⁻³). Confirmado por L = 128 vs
  200.
- τ máximo ≈ 2200–2600 MCS en t = +0.01…+0.02, B = 10⁻³ (el máximo de τ con
  campo chico está algo por encima de Tc). Tiempo: 2.2 h con 48 hilos (L = 200),
  1.25 h con 90 hilos (L = 128).
- t se define respecto del **Tc de Binder** (no del exacto): diferencia 4·10⁻⁵ en
  t, despreciable frente a |t|_min = 0.01.
- Límite inferior |t| = 0.01: ξ ≈ 0.57/|t| ≈ 57 sitios (T > Tc), L/ξ ≈ 3.5. Más
  cerca de Tc habría efectos de tamaño finito a campo bajo.

---

## 4. Cosas a revisar o evitar en las figuras

- `analysis/output/fss_data_collapse.png` (de `critical_analysis.jl`): el panel
  de Cv usa Cv/L^(α/ν) con el α/ν = 0.24 de la ley de potencia, que según (g) es
  aparente. Si querés colapso de Cv, el correcto para α = 0 es Cv − B ln L vs
  (T − Tc)L. El panel de χ está bien, pero usa el Tc de T*(L), no el de Binder.
- En los gráficos de χ en rango completo conviene escala log (si no, L = 200
  aplasta todo lo demás).
- La grilla de T no es uniforme: graficar con marcadores o líneas finas para que
  no parezca que hay más resolución en las colas.
- χ de L = 16 tiene un salto de ~30 % alrededor de T ≈ 1.96–2.03 (visible solo en
  escala log). Fluctuación de la varianza en red chica; no afecta ningún
  resultado.
- En (f), la rama B < 0 de `field_magnetization_curves.png` es reflejo por
  simetría, no datos. Si la usás, aclararlo.

## 5. Pendiente

- Problema 1 (derivaciones).
- El informe.
- Opcional: barras de error en χ, Cv, U (requiere volver a correr guardando
  muestras); interpolar el FWHM; corregir la selección del CSV de τ en
  `production_run.jl` (elegir el de más puntos).

## 6. Espacio en disco

El 2026-10-31 se movieron `timeseries_{mag,ene}_L{128,200}_nT96.csv` (~75 GB) y
`checkpoint_spins_L{128,200}_nT96.bin` a `~/ising_archivos_pesados/` (fuera del
proyecto, con un `LEEME.md`). Solo hacen falta para reanudar el estudio de
decorrelación con `resume_decorrelation_study.jl`, que los busca en
`run/data/`. Sin ellos el proyecto ocupa ~600 MB; lo más pesado que queda son
las curvas ρ(t) `run/data/autocorrelation_curves_L{128,200}_nT96.csv` (~540 MB),
que se conservan para poder graficar la autocorrelación.
