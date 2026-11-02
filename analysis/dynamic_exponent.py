"""
Exponente dinamico z a partir de los tiempos de autocorrelacion integrados
del estudio de decorrelacion (run/data/decorrelation_L*_nT96.csv).

Estimadores:
  1. tau_int en Tc: para cada L se ajusta un polinomio a ln tau_int en funcion
     de x = (T - Tc) L dentro de |x| <= a (a = 2, grado 2), descartando puntos
     a mas de 3 MAD; ln tau(Tc) = p(0). Error por bootstrap de residuos.
     Luego ln tau(Tc) = z ln L + c (ajuste ponderado).
  2. Colapso: z que minimiza la dispersion de ln(tau_int / L^z) entre tamanos
     en funcion de (T - Tc) L (nu = 1), con mediana movil de 3 puntos.
  3. Sistematicos: ventana a y grado del polinomio.

Salidas (analysis/output/):
  dynamic_exponent_z_tau_at_Tc.csv   tau_int(Tc) por L y observable
  dynamic_exponent_z_summary.csv     estimaciones de z
  dynamic_exponent_z.png / .pdf      figura de control (la del informe se
                                     genera en figuras/figuras_informe.ipynb)

Uso: python analysis/dynamic_exponent.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(ROOT, "run", "data")
OUT = os.path.join(ROOT, "analysis", "output")
os.makedirs(OUT, exist_ok=True)

Tc = 2 / np.log(1 + np.sqrt(2))
Ls = np.array([16, 32, 64, 128, 200])
rng = np.random.default_rng(1)

D = {L: pd.read_csv(os.path.join(DATA, f"decorrelation_L{L}_nT96.csv")) for L in Ls}


def tau_at_tc(L, col, a=2.0, deg=2, boot=0):
    """ln tau_int(Tc) por ajuste polinomial robusto en x = (T - Tc) L."""
    d = D[L]
    x = (d["T"].values - Tc) * L
    y = np.log(d[col].values)
    m = np.abs(x) <= a
    x, y = x[m], y[m]
    for _ in range(3):  # descarte iterativo de puntos a mas de 3 MAD
        c = np.polyfit(x, y, deg)
        r = y - np.polyval(c, x)
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        k = np.abs(r) < 3 * max(s, 1e-9)
        if k.all():
            break
        x, y = x[k], y[k]
    c = np.polyfit(x, y, deg)
    if not boot:
        return np.polyval(c, 0)
    r = y - np.polyval(c, x)
    v = [np.polyval(np.polyfit(x, np.polyval(c, x) + rng.choice(r, len(r)), deg), 0) for _ in range(boot)]
    return np.polyval(c, 0), np.std(v)


def fitz(lt, err=None, Lsel=Ls):
    """Ajuste lineal de ln tau vs ln L. Devuelve (z, error, chi2_red)."""
    A = np.vstack([np.log(Lsel), np.ones(len(Lsel))]).T
    w = 1 / err**2 if err is not None else np.ones(len(Lsel))
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    p = cov @ A.T @ W @ lt
    chi = np.sum(w * (lt - A @ p) ** 2) / (len(Lsel) - 2)
    scale = max(chi, 1) if err is not None else chi
    return p[0], np.sqrt(cov[0, 0] * scale), chi


def collapse_cost(z, col="tau_int_mag", xr=(-1.5, 4), Lsel=Ls):
    xs = np.linspace(*xr, 60)
    Y = []
    for L in Lsel:
        d = D[L]
        x = (d["T"].values - Tc) * L
        y = np.log(d[col].values) - z * np.log(L)
        ys = pd.Series(y).rolling(3, center=True, min_periods=1).median().values
        Y.append(interp1d(x, ys, bounds_error=False)(xs))
    return np.nanmean(np.nanvar(np.array(Y), axis=0))


# 1. tau_int(Tc) y ajuste de ley de potencia
res, rows = {}, []
for col, name in [("tau_int_mag", "abs_m"), ("tau_int_ene", "energy")]:
    out = [tau_at_tc(L, col, boot=400) for L in Ls]
    lt = np.array([o[0] for o in out])
    e = np.array([o[1] for o in out])
    res[col] = (lt, e)
    for L, a, b in zip(Ls, lt, e):
        rows.append(dict(observable=name, L=L, tau_int_Tc=round(np.exp(a), 1), rel_err=round(b, 3)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "dynamic_exponent_z_tau_at_Tc.csv"), index=False)

zM = fitz(*res["tau_int_mag"])
zM64 = fitz(res["tau_int_mag"][0][2:], res["tau_int_mag"][1][2:], Ls[2:])
zE = fitz(*res["tau_int_ene"])

# 2. colapso
zc = minimize_scalar(collapse_cost, bounds=(1.8, 2.5), method="bounded").x

# 3. sistematicos (ventana y grado)
sist = []
for a in [1.0, 1.5, 2.0, 3.0]:
    for deg in [1, 2, 3]:
        if deg >= 2 or a <= 1.5:
            lt = np.array([tau_at_tc(L, "tau_int_mag", a, deg) for L in Ls])
            sist += [fitz(lt)[0], fitz(lt[1:], None, Ls[1:])[0]]
zc_w = [minimize_scalar(lambda z: collapse_cost(z, xr=xr), bounds=(1.8, 2.5), method="bounded").x
        for xr in [(-1, 2), (-2, 4), (-1.5, 6), (0, 4)]]
sist += zc_w

summ = pd.DataFrame([
    dict(method="tau_int(|m|) at Tc, fit all L", z=round(zM[0], 3), err=round(zM[1], 3)),
    dict(method="tau_int(|m|) at Tc, L>=64", z=round(zM64[0], 3), err=round(zM64[1], 3)),
    dict(method="collapse of tau_int(|m|) vs (T-Tc)L", z=round(zc, 3), err=np.nan),
    dict(method="tau_int(e) at Tc (effective)", z=round(zE[0], 3), err=round(zE[1], 3)),
])
summ.to_csv(os.path.join(OUT, "dynamic_exponent_z_summary.csv"), index=False)

print(summ.to_string(index=False))
print(f"Rango de sistematicos (|m|): {min(sist):.3f} - {max(sist):.3f}")

# Figura de control
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(wspace=0.32))
cols = plt.cm.viridis(np.linspace(0, 0.85, len(Ls)))
for c, L in zip(cols, Ls):
    d = D[L]
    ax[0].plot((d["T"] - Tc) * L, d.tau_int_mag / L**zc, "o", ms=2.5, color=c, label=f"$L={L}$")
ax[0].set_yscale("log")
ax[0].set_xlim(-8, 8)
ax[0].set_xlabel(r"$(T-T_c)\,L$")
ax[0].set_ylabel(r"$\tau_{\mathrm{int}}(|m|)\,/\,L^{z}$")
ax[0].set_title(f"Colapso con $z = {zc:.2f}$", loc="left")
ax[0].legend(frameon=False, ncol=2)
LL = np.geomspace(13, 250)
for key, lab, mk, (z, ez, _), c in [("tau_int_mag", r"$|m|$", "o", zM, "#1f4e79"),
                                    ("tau_int_ene", r"$e$", "s", zE, "#b5651d")]:
    lt, e = res[key]
    ax[1].errorbar(Ls, np.exp(lt), yerr=np.exp(lt) * e, fmt=mk, ms=4, color=c, capsize=2)
    p = np.polyfit(np.log(Ls), lt, 1, w=1 / e)
    ax[1].plot(LL, np.exp(np.polyval(p, np.log(LL))), "-", color=c, lw=1, label=f"{lab}: {z:.2f}({round(ez * 100):d})")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"$L$")
ax[1].set_ylabel(r"$\tau_{\mathrm{int}}(T_c)$ [MCS]")
ax[1].legend(frameon=False)
fig.savefig(os.path.join(OUT, "dynamic_exponent_z.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "dynamic_exponent_z.pdf"), bbox_inches="tight")
