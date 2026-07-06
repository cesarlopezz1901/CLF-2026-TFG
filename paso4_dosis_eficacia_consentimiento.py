"""
=============================================================================
PASO 4 — ANÁLISIS DE DOSIS Y EFICACIA
TFG: Manejo del Paciente Paliativo en Sedación Continua — Hospital La Fe
=============================================================================
Solo pacientes con Consentimiento Informado (CI=1).
Genera 8 figuras PNG 180 dpi + 2 CSV de tablas.

Fuente de datos: resuelta por utils_io._resolver_paths()
  1. Variables de entorno EXCEL_1 / EXCEL_2
  2. config_datos.json
  3. Auto-detect por patrón en directorio de trabajo
=============================================================================
"""
import os
import sys

from utils_io import _resolver_paths, imprimir_configuracion

_E1_PATH, _E2_PATH, _FUENTE, _META = _resolver_paths()
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "resultados_paso4_CI")
os.makedirs(OUTPUT_DIR, exist_ok=True)

imprimir_configuracion("PASO 4 — CONFIGURACIÓN DE DATOS (CI=1)", _E1_PATH, _E2_PATH, _FUENTE, _META)


def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency, kruskal, fisher_exact

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    _HAS_LOWESS = True
except ImportError:
    _HAS_LOWESS = False

# ── Paleta y estilo (idéntica al PASO 3) ──────────────────────────────────────
C1      = "#C0392B"
C2      = "#1A5276"
CGLOBAL = "#2C3E50"
BG      = "#FAFAFA"
GRID_COLOR = "#E0E0E0"

ESCALA_FUENTE = 1.35  # Regla 1: factor global de escalado de fuente para legibilidad en PDF

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#CCCCCC", "axes.grid": True,
    "grid.color": GRID_COLOR, "grid.linestyle": "--", "grid.alpha": 0.6,
    "font.family": "DejaVu Sans", "font.size": 9 * ESCALA_FUENTE,
    "axes.titlesize": 10 * ESCALA_FUENTE, "axes.titleweight": "bold",
    "axes.labelsize": 9 * ESCALA_FUENTE, "xtick.labelsize": 8 * ESCALA_FUENTE, "ytick.labelsize": 8 * ESCALA_FUENTE,
    "legend.fontsize": 8 * ESCALA_FUENTE, "figure.dpi": 180,
})

LABEL_CASO = {1: "Caso 1 - Oncologico", 2: "Caso 2 - No Oncologico"}
COLOR_CASO = {1: C1, 2: C2}


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────────────────────

def _find_col(df, *keywords, require_all=False, exact=False):
    kws_l = [kw.lower() for kw in keywords]
    if exact:
        target = " ".join(kws_l)
        for col in df.columns:
            if col.strip().lower() == target:
                return col
    for col in df.columns:
        col_l = col.strip().lower()
        hits = [kw in col_l for kw in kws_l]
        if (all(hits) if require_all else any(hits)):
            return col
    return None


def _find_drug_col(df, *keywords):
    for col in df.columns:
        cl = col.lower()
        if all(kw.lower() in cl for kw in keywords):
            return col
    return None


def _norm_nhc(series):
    """Normaliza IDs: str, strip, lower, elimina .0 de floats."""
    return (
        series.astype(str).str.strip().str.lower()
        .str.replace(r"\.0$", "", regex=True)
    )


def _fmt_stat(test_name, stat_val):
    """OR=X.XX para Fisher, numérico para otros tests."""
    if pd.isna(stat_val) or stat_val is None:
        return "—"
    if test_name == "Fisher":
        if stat_val == np.inf:
            return "OR=inf"
        if stat_val == 0.0:
            return "OR=0"
        return f"OR={float(stat_val):.2f}"
    return f"{float(stat_val):.3f}"


# ── Helpers estadísticos ──────────────────────────────────────────────────────

def mw_test(g1, g2):
    g1c, g2c = g1.dropna(), g2.dropna()
    if len(g1c) < 2 or len(g2c) < 2:
        return np.nan, np.nan
    u, p = mannwhitneyu(g1c, g2c, alternative="two-sided")
    return u, p


def chi2_or_fisher(ct):
    """Chi² si expected >= 5 en todas las celdas, sino Fisher (estadístico = OR)."""
    ct = np.array(ct)
    if ct.shape == (2, 2):
        exp = chi2_contingency(ct)[3]
        if (exp < 5).any():
            _, p = fisher_exact(ct)
            a, b, c, d = ct[0, 0], ct[0, 1], ct[1, 0], ct[1, 1]
            if b * c > 0:
                or_val = round(float(a * d) / float(b * c), 4)
            elif a * d > 0:
                or_val = np.inf
            else:
                or_val = 0.0
            return "Fisher", or_val, p
    chi2, p, _, _ = chi2_contingency(ct)
    return "Chi2", chi2, p


def fmt_p(p):
    if pd.isna(p):   return "—"
    if p < 0.001:    return "<0.001**"
    if p < 0.01:     return f"{p:.3f}**"
    if p < 0.05:     return f"{p:.3f}*"
    return f"{p:.3f} ns"


def sig_label(p):
    if pd.isna(p):   return "ns"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
    return "ns"


def med_iqr(series):
    s = series.dropna()
    if len(s) == 0: return "—"
    return f"{s.median():.1f} [{s.quantile(.25):.1f}-{s.quantile(.75):.1f}]"


def r_mw(u, n1, n2):
    if pd.isna(u) or n1 == 0 or n2 == 0: return np.nan
    z = (u - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return abs(z) / np.sqrt(n1 + n2)


def prop_ic95(n, N):
    if N == 0: return 0, 0, 0
    p = n / N
    z = 1.96
    denom  = 1 + z**2 / N
    center = (p + z**2 / (2 * N)) / denom
    half   = z * np.sqrt(p * (1 - p) / N + z**2 / (4 * N**2)) / denom
    return p * 100, max(0, (center - half) * 100), min(100, (center + half) * 100)


def _combine_dt(date_ser, time_ser):
    """Combina fecha + hora. Si no hay hora, asume 12:00 (reduce sesgo)."""
    out = []
    for d, t in zip(date_ser, time_ser):
        if pd.isna(d):
            out.append(pd.NaT)
            continue
        try:
            base = pd.Timestamp(d)
            if pd.notna(t) and hasattr(t, "hour"):
                base = base.replace(hour=t.hour, minute=t.minute, second=0)
            elif pd.notna(t) and isinstance(t, str) and ":" in t:
                partes = t.strip().split(":")
                h = int(partes[0])
                m = int(partes[1]) if len(partes) > 1 else 0
                base = base.replace(hour=min(h, 23), minute=min(m, 59), second=0)
            elif pd.notna(t):
                frac = float(t)
                total_min = round(frac * 1440)
                base = base.replace(hour=min(total_min // 60, 23), minute=total_min % 60)
            else:
                base = base.replace(hour=12, minute=0, second=0)
        except Exception as e:
            print(f"  [AVISO] No se pudo parsear hora '{t}' para fecha {d}: {e}")
        out.append(base)
    return pd.Series(out, index=date_ser.index)


# ──────────────────────────────────────────────────────────────────────────────
# CARGA Y PREPARACIÓN DE DATOS
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    e1 = pd.read_excel(_E1_PATH)
    e2 = pd.read_excel(_E2_PATH)
    e1.columns = e1.columns.str.strip()
    e2.columns = e2.columns.str.strip()

    # ── columnas clave E1 ─────────────────────────────────────────────────────
    col_nhc_e1    = _find_col(e1, "nhc") or _find_col(e1, "historia") or _find_col(e1, "codigo")
    col_tipo      = _find_col(e1, "tipo", "paciente", require_all=True) or _find_col(e1, "tipo")
    col_ci        = _find_col(e1, "consentimiento") or next(
        (c for c in e1.columns if c.strip().lower() == "ci" or "consent" in c.lower()), None)
    col_fecha_ini  = _find_col(e1, "inicio", "sedaci", require_all=True) or _find_col(e1, "inicio")
    col_hora_ini   = _find_col(e1, "hora", "inicio", require_all=True)
    col_fecha_fall = (_find_col(e1, "fallecimiento") or _find_col(e1, "exitus")
                      or _find_col(e1, "muerte"))
    col_hora_fall  = (_find_col(e1, "hora", "xitus", require_all=True)
                      or _find_col(e1, "hora", "fall", require_all=True))
    col_dosis_bolo    = (_find_col(e1, "dosis bolo", exact=True)
                         or _find_col(e1, "dosis", "bolo", require_all=True))
    col_rescate_e1    = (_find_col(e1, "utilizacion", "rescate", require_all=True)
                         or _find_col(e1, "rescate"))
    col_dosis_resc_e1 = _find_col(e1, "dosis", "rescate", require_all=True)

    # ── normalizar E1 ─────────────────────────────────────────────────────────
    if col_nhc_e1 and col_nhc_e1 != "Nhc":
        e1["Nhc"] = e1[col_nhc_e1]
    e1["Nhc"] = _norm_nhc(e1["Nhc"])

    if col_tipo:
        e1["Tipo de paciente"] = e1[col_tipo]

    if col_fecha_ini:
        e1[col_fecha_ini] = pd.to_datetime(e1[col_fecha_ini], errors="coerce", dayfirst=True)
    if col_fecha_fall:
        e1[col_fecha_fall] = pd.to_datetime(e1[col_fecha_fall], errors="coerce", dayfirst=True)

    e1["ts_inicio_sed"] = _combine_dt(
        e1[col_fecha_ini] if col_fecha_ini else pd.Series([pd.NaT] * len(e1)),
        e1[col_hora_ini]  if col_hora_ini  else pd.Series([None]  * len(e1)),
    )
    n_con_hora = int((e1["ts_inicio_sed"].dt.hour != 0).sum())
    print(f"  [CHECK] {n_con_hora}/{len(e1)} pacientes con hora de inicio distinta de medianoche")
    e1["ts_fallec"] = _combine_dt(
        e1[col_fecha_fall] if col_fecha_fall else pd.Series([pd.NaT] * len(e1)),
        e1[col_hora_fall]  if col_hora_fall  else pd.Series([None]  * len(e1)),
    )

    e1["tiempo_fallec_dias"] = (
        (e1["ts_fallec"] - e1["ts_inicio_sed"]).dt.total_seconds() / 86400
    )

    # validación fechas éxitus
    neg_mask = e1["tiempo_fallec_dias"].notna() & (e1["tiempo_fallec_dias"] < 0)
    if neg_mask.any():
        bad = list(e1.loc[neg_mask, "Nhc"])
        print(f"  [WARN] {neg_mask.sum()} paciente(s) con exitus ANTES del inicio (negativo) -> NaN: {bad}")
        e1.loc[neg_mask, "tiempo_fallec_dias"] = np.nan
    largo_mask = e1["tiempo_fallec_dias"].notna() & (e1["tiempo_fallec_dias"] > 30)
    if largo_mask.any():
        bad = list(e1.loc[largo_mask, "Nhc"])
        print(f"  [WARN] {largo_mask.sum()} paciente(s) con supervivencia >30 dias (verificar): {bad}")

    e1["dosis_bolo_mid_e1"] = pd.to_numeric(
        e1[col_dosis_bolo] if col_dosis_bolo else pd.Series([np.nan] * len(e1)), errors="coerce")
    e1["uso_rescate"] = pd.to_numeric(
        e1[col_rescate_e1] if col_rescate_e1 else pd.Series([np.nan] * len(e1)), errors="coerce")
    e1["dosis_resc_total_e1"] = pd.to_numeric(
        e1[col_dosis_resc_e1] if col_dosis_resc_e1 else pd.Series([np.nan] * len(e1)), errors="coerce")

    # ── columnas clave E2 ─────────────────────────────────────────────────────
    col_nhc_e2   = _find_col(e2, "nhc") or _find_col(e2, "historia") or _find_col(e2, "codigo")
    col_ramsay   = _find_col(e2, "ramsay")
    col_fvis     = _find_col(e2, "fecha", "visita", require_all=True) or _find_col(e2, "fecha")
    col_hvis     = _find_col(e2, "hora", "visita", require_all=True) or _find_col(e2, "hora")
    col_mid_e2   = _find_drug_col(e2, "midazolam", "infusor") or _find_drug_col(e2, "midazolam")
    col_resc_e2  = _find_col(e2, "rescate")
    col_dosis_re2 = _find_col(e2, "dosis", "rescate", require_all=True)

    if col_nhc_e2 and col_nhc_e2 != "Nhc":
        e2["Nhc"] = e2[col_nhc_e2]
    e2["Nhc"] = _norm_nhc(e2["Nhc"])

    if col_ramsay:
        e2["Escala Ramsay"] = pd.to_numeric(e2[col_ramsay], errors="coerce")
    if col_fvis:
        e2[col_fvis] = pd.to_datetime(e2[col_fvis], errors="coerce", dayfirst=True)
    if col_mid_e2:
        e2[col_mid_e2] = pd.to_numeric(e2[col_mid_e2], errors="coerce")

    e2["ts_visita"] = _combine_dt(
        e2[col_fvis] if col_fvis else pd.Series([pd.NaT] * len(e2)),
        e2[col_hvis] if col_hvis else pd.Series([None]  * len(e2)),
    )
    e2 = e2.sort_values(["Nhc", "ts_visita"]).reset_index(drop=True)

    e2["dosis_mid_visita"] = pd.to_numeric(
        e2[col_mid_e2] if col_mid_e2 else pd.Series([np.nan] * len(e2)), errors="coerce")

    if col_dosis_re2:
        e2["dosis_resc_visita"] = pd.to_numeric(e2[col_dosis_re2], errors="coerce")
    elif col_resc_e2:
        e2["dosis_resc_visita"] = pd.to_numeric(e2[col_resc_e2], errors="coerce")
    else:
        e2["dosis_resc_visita"] = np.nan

    if col_resc_e2:
        e2["uso_resc_visita"] = pd.to_numeric(e2[col_resc_e2], errors="coerce").gt(0).astype(float)
    else:
        e2["uso_resc_visita"] = np.nan

    # ── validación de merge ───────────────────────────────────────────────────
    nhc_e1     = set(e1["Nhc"].dropna())
    nhc_e2     = set(e2["Nhc"].dropna())
    matched    = nhc_e1 & nhc_e2
    sin_vis_e2 = nhc_e1 - nhc_e2
    sin_e1     = nhc_e2 - nhc_e1
    print("=== VALIDACION DE MERGE ===")
    print(f"Pacientes en E1: {len(nhc_e1)} | Unicos en E2: {len(nhc_e2)} | "
          f"Match exitoso: {len(matched)} | Sin visitas E2: {len(sin_vis_e2)}")
    if sin_vis_e2:
        print(f"  [WARN] IDs de E1 sin visitas en E2: {sorted(sin_vis_e2)}")
    if sin_e1:
        print(f"  [WARN] IDs de E2 sin match en E1: {sorted(sin_e1)}")

    # ── agregados por paciente desde E2 ──────────────────────────────────────
    def _agg_e2(e2_df, e1_nhcs):
        rows = []
        for nhc, grp in e2_df[e2_df["Nhc"].isin(e1_nhcs)].groupby("Nhc"):
            grp = grp.sort_values("ts_visita")

            dosis_acum  = grp["dosis_mid_visita"].sum(min_count=1)
            ram_primera = grp["Escala Ramsay"].dropna().iloc[0] if grp["Escala Ramsay"].notna().any() else np.nan
            ram_ultima  = grp["Escala Ramsay"].dropna().iloc[-1] if grp["Escala Ramsay"].notna().any() else np.nan
            ts_primera  = grp["ts_visita"].iloc[0]
            ts_ultima   = grp["ts_visita"].iloc[-1]

            ram6 = grp[grp["Escala Ramsay"] == 6]
            if len(ram6):
                ts_pram6      = ram6["ts_visita"].iloc[0]
                dosis_pram6   = ram6["dosis_mid_visita"].iloc[0]
                ramsay_pram6  = ram6["Escala Ramsay"].iloc[0]
            else:
                ts_pram6      = pd.NaT
                dosis_pram6   = np.nan
                ramsay_pram6  = np.nan

            ram6_mant  = grp[grp["Escala Ramsay"] == 6].copy()
            mant_h = np.nan
            if len(ram6_mant) >= 2:
                deltas = ram6_mant["ts_visita"].diff().dt.total_seconds().dropna() / 3600
                mant_h = float(deltas.sum())

            n_resc     = int(grp["uso_resc_visita"].fillna(0).gt(0).sum())
            dosis_resc = grp["dosis_resc_visita"].sum(min_count=1)

            rows.append({
                "Nhc":               nhc,
                "dosis_acum_mid":    float(dosis_acum)    if not pd.isna(dosis_acum)   else np.nan,
                "ramsay_primera":    float(ram_primera)   if not pd.isna(ram_primera)  else np.nan,
                "ramsay_ultima":     float(ram_ultima)    if not pd.isna(ram_ultima)   else np.nan,
                "ts_primera_vis":    ts_primera,
                "ts_ultima_vis":     ts_ultima,
                "ts_primera_ram6":   ts_pram6,
                "dosis_primera_ram6":  float(dosis_pram6)  if not pd.isna(dosis_pram6)  else np.nan,
                "ramsay_primera_ram6": float(ramsay_pram6) if not pd.isna(ramsay_pram6) else np.nan,
                "tiempo_mant_h":     mant_h,
                "n_rescates_e2":     n_resc,
                "dosis_resc_e2":     float(dosis_resc) if not pd.isna(dosis_resc) else np.nan,
            })
        return pd.DataFrame(rows)

    agg = _agg_e2(e2, nhc_e1)
    e1  = e1.merge(agg, on="Nhc", how="left")

    e1["tiempo_hasta_ram6_h"] = (
        (e1["ts_primera_ram6"] - e1["ts_inicio_sed"]).dt.total_seconds() / 3600
    )

    # auditoría tiempo_hasta_ram6_h
    audit = e1[["Nhc", "ts_inicio_sed", "ts_primera_ram6", "tiempo_hasta_ram6_h"]].dropna()
    neg_h = (e1["tiempo_hasta_ram6_h"] < 0).sum()
    if neg_h:
        print(f"  [WARN] {neg_h} paciente(s) con tiempo_hasta_ram6_h negativo")
    print(f"  Auditoria tiempo_hasta_ram5 [{len(audit)} pacientes con Ramsay=6 antes del filtro CI]:")
    for _, r in audit.head(8).iterrows():
        ini_str = r["ts_inicio_sed"].strftime("%d/%m %H:%M") if pd.notna(r["ts_inicio_sed"]) else "NaT"
        r5_str  = r["ts_primera_ram6"].strftime("%d/%m %H:%M") if pd.notna(r["ts_primera_ram6"]) else "NaT"
        print(f"    {r['Nhc']}: inicio={ini_str} | ram6={r5_str} | h={r['tiempo_hasta_ram6_h']:.1f}")
    if len(audit) > 8:
        print(f"    ... ({len(audit)-8} mas no mostrados)")

    # ── FILTRO CONSENTIMIENTO INFORMADO ───────────────────────────────────────
    if col_ci:
        ci_vals = pd.to_numeric(e1[col_ci], errors="coerce")
        n_total = len(e1)
        e1 = e1[ci_vals == 1].reset_index(drop=True)
        print(f"   [CI] {len(e1)} pacientes con CI=1 ({n_total - len(e1)} excluidos)")
        nhc_ci = set(e1["Nhc"].dropna())
        e2 = e2[e2["Nhc"].isin(nhc_ci)].reset_index(drop=True)
    else:
        print("   [CI] AVISO: columna CI no encontrada — se incluyen todos los pacientes")
    # ─────────────────────────────────────────────────────────────────────────

    e1["tipo_label"] = e1["Tipo de paciente"].map(LABEL_CASO)

    print(f"   E1: {len(e1)} pacientes | E2: {len(e2)} visitas")
    print(f"   Dosis acum. midazolam: {e1['dosis_acum_mid'].notna().sum()} validos "
          f"| mediana={e1['dosis_acum_mid'].median():.1f} mg")
    print(f"   Ramsay = 6 alcanzado: {e1['ts_primera_ram6'].notna().sum()} pacientes")

    return e1, e2


# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 1 — RELACIÓN DOSIS-RESPUESTA GLOBAL (2 paneles)
# ──────────────────────────────────────────────────────────────────────────────

def fig1_dosis_respuesta(df):
    print("Generando Fig1 — Dosis-Respuesta Global (2 paneles)...")

    valid_a = df[["dosis_acum_mid", "ramsay_ultima", "Tipo de paciente"]].dropna()
    valid_b = df[["dosis_primera_ram6", "ramsay_primera_ram6", "Tipo de paciente"]].dropna()

    if len(valid_a) < 5 and len(valid_b) < 3:
        _fig_vacia("P4_Fig1_Dosis_Respuesta_Global.png", "Fig1: datos insuficientes")
        return {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5))
    fig.suptitle("FIGURA 1 - Relacion Dosis-Respuesta (solo CI=1)",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)
    stats_table = {}

    def _plot_scatter(ax, xdata, ydata, tipos, xlabel, ylabel, panel_label):
        rho, pval = (stats.spearmanr(xdata, ydata) if len(xdata) >= 3
                     else (np.nan, np.nan))
        for t, lbl in LABEL_CASO.items():
            mask = tipos == t
            ax.scatter(xdata[mask], ydata[mask], c=COLOR_CASO[t], label=lbl,
                       s=60, alpha=0.75, edgecolors="white", linewidths=0.5, zorder=3)

        if len(xdata) >= 10 and _HAS_LOWESS:
            order  = np.argsort(xdata)
            xs, ys = xdata[order], ydata[order]
            smooth = sm_lowess(ys, xs, frac=0.6, return_sorted=True)
            x_grid = np.linspace(xs.min(), xs.max(), 200)
            ax.plot(smooth[:, 0], smooth[:, 1], color=CGLOBAL, lw=2.2,
                    label="LOWESS", zorder=4)
            np.random.seed(42)
            boot = []
            for _ in range(200):
                idx = np.random.choice(len(xs), len(xs), replace=True)
                b = sm_lowess(ys[idx], xs[idx], frac=0.6, return_sorted=True)
                if len(b) > 1:
                    boot.append(np.interp(x_grid, b[:, 0], b[:, 1]))
            if boot:
                ax.fill_between(x_grid, np.percentile(boot, 2.5, axis=0),
                                np.percentile(boot, 97.5, axis=0),
                                color=CGLOBAL, alpha=0.15, label="IC 95%")
        elif len(xdata) >= 3:
            slope, intercept, *_ = stats.linregress(xdata, ydata)
            x_fit = np.linspace(xdata.min(), xdata.max(), 200)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, color=CGLOBAL, lw=2, label="Regresion lineal", zorder=4)
            np.random.seed(42)
            boot_y = []
            for _ in range(200):
                idx = np.random.choice(len(xdata), len(xdata), replace=True)
                try:
                    s2, i2 = np.polyfit(xdata[idx], ydata[idx], 1)
                    boot_y.append(s2 * x_fit + i2)
                except Exception:
                    pass
            if boot_y:
                ax.fill_between(x_fit, np.percentile(boot_y, 2.5, axis=0),
                                np.percentile(boot_y, 97.5, axis=0),
                                color=CGLOBAL, alpha=0.15, label="IC 95%")

        ax.set_xlabel(xlabel, fontsize=9 * ESCALA_FUENTE)
        ax.set_ylabel(ylabel, fontsize=9 * ESCALA_FUENTE)
        ax.set_yticks([1, 2, 3, 4, 5, 6])
        ax.axhline(6, color="#E74C3C", ls="--", lw=1.2, alpha=0.7,
                   label="Ramsay = 6 (objetivo)")
        rho_s = f"{rho:.3f}" if not pd.isna(rho) else "n/c"
        ax.set_title(f"{panel_label}\nSpearman r={rho_s}, p={fmt_p(pval)}  (n={len(xdata)})",
                     fontsize=9 * ESCALA_FUENTE, fontweight="bold")
        ax.legend(fontsize=7.5 * ESCALA_FUENTE, framealpha=0.9)
        return {"test": "Spearman",
                "stat": round(rho, 4) if not pd.isna(rho) else np.nan,
                "p":    round(pval, 4) if not pd.isna(pval) else np.nan,
                "sig":  sig_label(pval)}

    if len(valid_a) >= 5:
        st = _plot_scatter(
            axes[0],
            valid_a["dosis_acum_mid"].values,
            valid_a["ramsay_ultima"].values,
            valid_a["Tipo de paciente"].values,
            "Dosis total acumulada de Midazolam (mg)",
            "Escala Ramsay — ultima visita",
            "1A - Dosis Acumulada vs Ramsay Final",
        )
        axes[0].text(0.02, 0.02,
                     "Nota: dosis acumulada esta influenciada\npor tiempo de supervivencia",
                     transform=axes[0].transAxes, fontsize=7 * ESCALA_FUENTE, color="gray",
                     bbox=dict(boxstyle="round", facecolor="#FFF9C4",
                               edgecolor="#F9A825", alpha=0.85))
        stats_table["1A Spearman dosis-acum vs Ramsay"] = st
    else:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "Datos insuficientes (n<5)", ha="center",
                     va="center", transform=axes[0].transAxes, fontsize=9 * ESCALA_FUENTE, color="gray")

    if len(valid_b) >= 3:
        st = _plot_scatter(
            axes[1],
            valid_b["dosis_primera_ram6"].values,
            valid_b["ramsay_primera_ram6"].values,
            valid_b["Tipo de paciente"].values,
            "Dosis Minima Eficaz — primera visita con Ramsay = 6 (mg)",
            "Escala Ramsay en esa visita (5 o 6)",
            "1B - Dosis Minima Eficaz vs Ramsay\n(relacion clinicamente valida)",
        )
        stats_table["1B Spearman dosis-minima vs Ramsay"] = st
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Datos insuficientes (n<3)", ha="center",
                     va="center", transform=axes[1].transAxes, fontsize=9 * ESCALA_FUENTE, color="gray")

    fig.tight_layout(rect=[0, 0, 1, 0.90], w_pad=2.8)
    fig.savefig(out("P4_Fig1_Dosis_Respuesta_Global.png"), dpi=180,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig1 guardada")
    return stats_table


# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 2 — PROPORCIÓN SEDACIÓN PROFUNDA (calculada directamente desde E2)
# ──────────────────────────────────────────────────────────────────────────────

def fig2_proporcion_sedacion(df, e2):
    print("Generando Fig2 — Proporcion Sedacion Profunda...")
    tipo_map = df.set_index("Nhc")["Tipo de paciente"].dropna().to_dict()
    stats_table = {}

    registros = []
    for nhc, grp in e2.groupby("Nhc"):
        t = tipo_map.get(nhc)
        if t is None:
            continue
        grp_s    = grp.sort_values("ts_visita")
        ram_vals = grp_s["Escala Ramsay"].dropna()
        if len(ram_vals) == 0:
            continue
        registros.append({
            "nhc":         nhc,
            "tipo":        t,
            "ram_primera": float(ram_vals.iloc[0]),
            "ram_ultima":  float(ram_vals.iloc[-1]),
            "n_visitas":   len(grp_s),
        })

    if not registros:
        _fig_vacia("P4_Fig2_Proporcion_Sedacion_Profunda.png",
                   "Fig2: sin datos E2 con Ramsay valido")
        return {}

    reg_df = pd.DataFrame(registros)
    misma  = (reg_df["ram_primera"] == reg_df["ram_ultima"]).sum()
    if misma == len(reg_df):
        nota_vis = "* todos los pacientes tienen primera=ultima visita"
    elif misma > 0:
        nota_vis = f"* {misma} paciente(s) con una sola visita (primera=ultima)"
    else:
        nota_vis = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5))
    fig.suptitle(
        "FIGURA 2 - Proporcion que Alcanza Sedacion Profunda (Ramsay = 6) [CI=1]\n"
        "Primera visita vs. Ultima visita por Tipo de Paciente",
        fontsize=11 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL,
    )

    for ax, col_r, titulo, moment in [
        (axes[0], "ram_primera", "2A - Primera visita de seguimiento", "Primera visita"),
        (axes[1], "ram_ultima",  "2B - Ultima visita de seguimiento",  "Ultima visita"),
    ]:
        datos = []
        for t in (1, 2):
            sub    = reg_df[reg_df["tipo"] == t][col_r].dropna()
            n_tot  = len(sub)
            n_ram6 = int((sub == 6).sum())
            pct, ic_lo, ic_hi = prop_ic95(n_ram6, n_tot)
            datos.append((t, n_ram6, n_tot, pct, ic_lo, ic_hi))

        if datos[0][2] > 0 and datos[1][2] > 0:
            ct = np.array([
                [datos[0][1], datos[0][2] - datos[0][1]],
                [datos[1][1], datos[1][2] - datos[1][1]],
            ])
            try:
                test_name, chi2v, pv = chi2_or_fisher(ct)
            except Exception:
                test_name, chi2v, pv = "—", np.nan, np.nan
        else:
            test_name, chi2v, pv = "—", np.nan, np.nan

        xs      = np.arange(2)
        pcts    = [d[3] for d in datos]
        ic_lo2  = [d[4] for d in datos]
        ic_hi2  = [d[5] for d in datos]
        e_lo    = [p - l for p, l in zip(pcts, ic_lo2)]
        e_hi    = [h - p for p, h in zip(pcts, ic_hi2)]

        bars = ax.bar(xs, pcts, color=[C1, C2], alpha=0.82, edgecolor="white", width=0.5)
        ax.errorbar(xs, pcts, yerr=[e_lo, e_hi],
                    fmt="none", color=CGLOBAL, capsize=5, lw=1.5)
        for i, (b, d) in enumerate(zip(bars, datos)):
            # Etiqueta situada por encima del extremo superior del IC95%
            # (d[5] = ic_hi) para no cruzarse con la barra de error (Regla 6).
            label_y = d[5] + 6
            ax.text(b.get_x() + b.get_width() / 2, label_y,
                    f"{d[3]:.1f}%\n(n={d[1]}/{d[2]})", ha="center", va="bottom", fontsize=8 * ESCALA_FUENTE)

        ax.set_xticks(xs)
        ax.set_xticklabels([LABEL_CASO[1], LABEL_CASO[2]], fontsize=8 * ESCALA_FUENTE)
        ax.set_ylabel("% pacientes con Ramsay = 6", fontsize=9 * ESCALA_FUENTE)
        ax.set_ylim(0, 135)
        stat_str = _fmt_stat(test_name, chi2v)
        ax.set_title(f"{titulo}\n{test_name} ({stat_str}) p={fmt_p(pv)}", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
        if nota_vis:
            ax.text(0.02, 0.98, nota_vis, transform=ax.transAxes, fontsize=7 * ESCALA_FUENTE,
                    va="top", color="gray")

        stats_table[f"Ramsay=6 {moment}"] = {
            "test": test_name,
            "stat": chi2v,
            "p":    round(pv, 4) if not pd.isna(pv) else np.nan,
            "sig":  sig_label(pv),
        }

    fig.tight_layout(rect=[0, 0, 1, 0.88], w_pad=2.8)
    fig.savefig(out("P4_Fig2_Proporcion_Sedacion_Profunda.png"), dpi=180,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig2 guardada")
    return stats_table


# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 3 — DOSIS MÍNIMA EFICAZ
# ──────────────────────────────────────────────────────────────────────────────

def fig3_dosis_minima_eficaz(df):
    print("Generando Fig3 — Dosis Minima Eficaz...")
    valid = df[["dosis_primera_ram6", "Tipo de paciente"]].dropna()
    g1 = valid[valid["Tipo de paciente"] == 1]["dosis_primera_ram6"]
    g2 = valid[valid["Tipo de paciente"] == 2]["dosis_primera_ram6"]

    if len(g1) < 3 and len(g2) < 3:
        _fig_vacia("P4_Fig3_Dosis_Minima_Eficaz.png", "Fig3: datos insuficientes")
        return {}

    u, pval = mw_test(g1, g2)
    r_ef    = r_mw(u, len(g1), len(g2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(
        "FIGURA 3 - Dosis Minima Eficaz de Midazolam [CI=1]\n"
        "(dosis en infusor en la primera visita con Ramsay = 6)",
        fontsize=11 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL,
    )

    data_bp = [g1.values, g2.values]
    bp = ax1.boxplot(data_bp, patch_artist=True, widths=0.4,
                     medianprops=dict(color="white", lw=2),
                     whiskerprops=dict(color=CGLOBAL), capprops=dict(color=CGLOBAL),
                     flierprops=dict(marker="o", markersize=4, alpha=0.4))
    for patch, color in zip(bp["boxes"], [C1, C2]):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    for i, (g, c) in enumerate([(g1, C1), (g2, C2)], 1):
        jitter = np.random.uniform(-0.15, 0.15, size=len(g))
        ax1.scatter(np.full(len(g), i) + jitter, g.values, color=c,
                    alpha=0.55, s=35, zorder=3, edgecolors="white", lw=0.4)
    # Techo comun para dejar margen claro por debajo del titulo (Regla 7):
    # las etiquetas de mediana se colocan justo encima del maximo de cada
    # grupo, y el limite superior del eje se sube para que no choquen con
    # el titulo de dos lineas del panel.
    max_global = max(g1.max() if len(g1) else 0, g2.max() if len(g2) else 0)
    ax1.set_ylim(top=max_global * 1.42)
    for i, (g, c) in enumerate([(g1, C1), (g2, C2)], 1):
        if len(g):
            ax1.text(i, g.max() * 1.10,
                     f"Med={g.median():.1f}\n[{g.quantile(.25):.1f}-{g.quantile(.75):.1f}]",
                     ha="center", va="bottom", fontsize=7.5 * ESCALA_FUENTE, color=c, fontweight="bold")
    ax1.set_xticks([1, 2]); ax1.set_xticklabels([LABEL_CASO[1], LABEL_CASO[2]])
    ax1.set_ylabel("Dosis Midazolam (mg)", fontsize=9 * ESCALA_FUENTE)
    r_ef_s = f"{r_ef:.2f}" if not pd.isna(r_ef) else "n/c"
    ax1.set_title(f"Boxplot + Jitter\nMann-Whitney U={u:.0f}, p={fmt_p(pval)}, r={r_ef_s}",
                  fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    for g, c, lbl in [(g1, C1, LABEL_CASO[1]), (g2, C2, LABEL_CASO[2])]:
        if len(g) < 2: continue
        qs   = [0, 10, 25, 50, 75, 90, 100]
        vals = np.percentile(g.dropna(), qs)
        ax2.plot(qs, vals, "o-", color=c, label=lbl, lw=2, markersize=5, alpha=0.85)
        ax2.fill_between([25, 75], [vals[2], vals[2]], [vals[4], vals[4]],
                         color=c, alpha=0.12)
    ax2.set_xlabel("Percentil", fontsize=9 * ESCALA_FUENTE)
    ax2.set_ylabel("Dosis Midazolam (mg)", fontsize=9 * ESCALA_FUENTE)
    ax2.set_title("Distribucion en Percentiles (P10-P90)", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
    ax2.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.88], w_pad=2.8)
    fig.savefig(out("P4_Fig3_Dosis_Minima_Eficaz.png"), dpi=180,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig3 guardada")
    return {"Dosis minima eficaz (MW)": {
        "test": "Mann-Whitney U",
        "stat": round(u, 2) if not pd.isna(u) else np.nan,
        "p":    round(pval, 4) if not pd.isna(pval) else np.nan,
        "sig":  sig_label(pval),
    }}


# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 4 — TIEMPOS CLÍNICOS
# ──────────────────────────────────────────────────────────────────────────────

def fig4_tiempos_clinicos(df):
    print("Generando Fig4 — Tiempos Clinicos...")
    stats_table = {}

    metricas = [
        ("tiempo_hasta_ram6_h",  "a) Tiempo hasta Ramsay = 6 (h)",       "Horas"),
        ("tiempo_mant_h",         "b) Mantenimiento sedacion profunda (h)", "Horas"),
        ("tiempo_fallec_dias",    "c) Tiempo inicio sedacion exitus (dias)", "Dias"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 8))
    fig.suptitle("FIGURA 4 - Tiempos Clinicos del Proceso de Sedacion [CI=1]",
                 fontsize=11 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)

    for ax, (col, titulo_base, unidad) in zip(axes, metricas):
        g1 = df[df["Tipo de paciente"] == 1][col].dropna()
        g2 = df[df["Tipo de paciente"] == 2][col].dropna()

        if len(g1) < 2 and len(g2) < 2:
            ax.text(0.5, 0.5, "Datos insuficientes", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9 * ESCALA_FUENTE, color="gray")
            ax.set_title(titulo_base, fontsize=9 * ESCALA_FUENTE, fontweight="bold")
            continue

        u, pval = mw_test(g1, g2)
        r_ef    = r_mw(u, len(g1), len(g2))

        all_vals = pd.concat([g1, g2])
        if len(all_vals) > 0:
            mode_cnt = all_vals.value_counts().iloc[0]
            tie_pct  = mode_cnt / len(all_vals)
            empate_nota = f"\n[Empates: {tie_pct:.0%} valores identicos]" if tie_pct > 0.5 else ""
        else:
            empate_nota = ""

        data_bp   = [g.values for g in (g1, g2) if len(g) >= 1]
        positions = [i + 1 for i, g in enumerate((g1, g2)) if len(g) >= 1]
        colors_bp = [c for c, g in zip((C1, C2), (g1, g2)) if len(g) >= 1]

        bp = ax.boxplot(data_bp, positions=positions, patch_artist=True, widths=0.4,
                        medianprops=dict(color="white", lw=2),
                        whiskerprops=dict(color=CGLOBAL), capprops=dict(color=CGLOBAL),
                        flierprops=dict(marker="o", markersize=3, alpha=0.4))
        for patch, color in zip(bp["boxes"], colors_bp):
            patch.set_facecolor(color); patch.set_alpha(0.75)

        for i, (g, c) in enumerate([(g1, C1), (g2, C2)], 1):
            if len(g) < 1: continue
            jitter = np.random.uniform(-0.13, 0.13, size=len(g))
            ax.scatter(np.full(len(g), i) + jitter, g.values, color=c,
                       alpha=0.5, s=30, zorder=3, edgecolors="white", lw=0.3)
            ax.text(i, float(g.max()) * 1.05 + 0.1, f"Med={g.median():.1f}",
                    ha="center", va="bottom", fontsize=7.5 * ESCALA_FUENTE, color=c, fontweight="bold")

        # Techo con margen amplio: el titulo del panel ocupa 3 lineas y las
        # etiquetas de mediana quedaban pegadas a el si el eje no dejaba aire.
        max_g = max([float(g.max()) for g in (g1, g2) if len(g) >= 1], default=1)
        ax.set_ylim(top=max_g * 1.3 + 0.5)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([LABEL_CASO[1].replace(" - ", "\n"),
                             LABEL_CASO[2].replace(" - ", "\n")], fontsize=8 * ESCALA_FUENTE)
        ax.set_ylabel(unidad, fontsize=9 * ESCALA_FUENTE)
        r_s = f"{r_ef:.2f}" if not pd.isna(r_ef) else "n/c"
        titulo = f"{titulo_base}\nMW p={fmt_p(pval)}, r={r_s}{empate_nota}"
        ax.set_title(titulo, fontsize=8.5 * ESCALA_FUENTE, fontweight="bold")
        stats_table[titulo_base] = {
            "test": "Mann-Whitney U",
            "stat": round(u, 2) if not pd.isna(u) else np.nan,
            "p":    round(pval, 4) if not pd.isna(pval) else np.nan,
            "sig":  sig_label(pval),
        }

    fig.tight_layout(w_pad=2.8)
    fig.savefig(out("P4_Fig4_Tiempos_Clinicos.png"), dpi=180,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig4 guardada")
    return stats_table


# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 5 — GESTIÓN CLÍNICA: RESCATES Y ESCALADA DE DOSIS
# ──────────────────────────────────────────────────────────────────────────────

def fig5_gestion_rescates(df, e2):
    print("Generando Fig5 — Gestion de Rescates...")
    stats_table = {}
    g1 = df[df["Tipo de paciente"] == 1]
    g2 = df[df["Tipo de paciente"] == 2]

    fig = plt.figure(figsize=(13, 14))
    fig.suptitle("FIGURA 5 - Gestion Clinica: Rescates y Escalada de Dosis [CI=1]",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL, y=0.99)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.75, wspace=0.5)

    ax5a = fig.add_subplot(gs[0, 0])
    pcts_r, ns_r, ns_tot = [], [], []
    for grp in (g1, g2):
        serie = pd.to_numeric(grp["uso_rescate"], errors="coerce").dropna()
        n_tot = len(serie); n_si = int((serie > 0).sum())
        pct, _, _ = prop_ic95(n_si, n_tot)
        pcts_r.append(pct); ns_r.append(n_si); ns_tot.append(n_tot)
    bars = ax5a.bar([0, 1], pcts_r, color=[C1, C2], alpha=0.82, edgecolor="white", width=0.5)
    for i, (b, n, N) in enumerate(zip(bars, ns_r, ns_tot)):
        ax5a.text(b.get_x() + b.get_width() / 2, pcts_r[i] + 1.5,
                  f"{pcts_r[i]:.1f}%\n(n={n}/{N})", ha="center", fontsize=8 * ESCALA_FUENTE)
    try:
        ct = np.array([[ns_r[0], ns_tot[0]-ns_r[0]], [ns_r[1], ns_tot[1]-ns_r[1]]])
        tn, tv, pv = chi2_or_fisher(ct)
    except Exception:
        tn, tv, pv = "—", np.nan, np.nan
    ax5a.set_xticks([0, 1]); ax5a.set_xticklabels([LABEL_CASO[1], LABEL_CASO[2]], fontsize=8 * ESCALA_FUENTE)
    ax5a.set_ylabel("% pacientes con >= 1 rescate", fontsize=9 * ESCALA_FUENTE)
    ax5a.set_ylim(0, 115)
    ax5a.set_title(f"5A - Uso de Rescate\n{tn} ({_fmt_stat(tn, tv)}) p={fmt_p(pv)}",
                   fontsize=9 * ESCALA_FUENTE, fontweight="bold")
    stats_table["Uso rescate (%)"] = {"test": tn, "stat": tv,
                                       "p": round(pv, 4) if not pd.isna(pv) else np.nan,
                                       "sig": sig_label(pv)}

    ax5b = fig.add_subplot(gs[0, 1])
    nr1 = g1["n_rescates_e2"].dropna(); nr2 = g2["n_rescates_e2"].dropna()
    u, pv = mw_test(nr1, nr2)
    for i, (g, c) in enumerate([(nr1, C1), (nr2, C2)], 1):
        if len(g) < 1: continue
        bp = ax5b.boxplot(g.values, positions=[i], patch_artist=True, widths=0.4,
                          medianprops=dict(color="white", lw=2),
                          whiskerprops=dict(color=CGLOBAL), capprops=dict(color=CGLOBAL))
        bp["boxes"][0].set_facecolor(c); bp["boxes"][0].set_alpha(0.75)
        jitter = np.random.uniform(-0.13, 0.13, size=len(g))
        ax5b.scatter(np.full(len(g), i) + jitter, g.values, color=c, alpha=0.5, s=30, zorder=3)
    ax5b.set_xticks([1, 2]); ax5b.set_xticklabels([LABEL_CASO[1], LABEL_CASO[2]], fontsize=8 * ESCALA_FUENTE)
    ax5b.set_ylabel("Numero de rescates", fontsize=9 * ESCALA_FUENTE)
    ax5b.set_title(f"5B - Numero de Rescates\nMW p={fmt_p(pv)}", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
    stats_table["N rescates (MW)"] = {"test": "Mann-Whitney U",
                                       "stat": round(u, 2) if not pd.isna(u) else np.nan,
                                       "p": round(pv, 4) if not pd.isna(pv) else np.nan,
                                       "sig": sig_label(pv)}

    ax5c = fig.add_subplot(gs[1, 0])
    dr1 = g1["dosis_resc_total_e1"].dropna()
    dr2 = g2["dosis_resc_total_e1"].dropna()
    if len(dr1) == 0: dr1 = g1["dosis_resc_e2"].dropna()
    if len(dr2) == 0: dr2 = g2["dosis_resc_e2"].dropna()
    u, pv = mw_test(dr1, dr2)
    for i, (g, c) in enumerate([(dr1, C1), (dr2, C2)], 1):
        if len(g) < 1: continue
        bp = ax5c.boxplot(g.values, positions=[i], patch_artist=True, widths=0.4,
                          medianprops=dict(color="white", lw=2),
                          whiskerprops=dict(color=CGLOBAL), capprops=dict(color=CGLOBAL))
        bp["boxes"][0].set_facecolor(c); bp["boxes"][0].set_alpha(0.75)
        jitter = np.random.uniform(-0.13, 0.13, size=len(g))
        ax5c.scatter(np.full(len(g), i) + jitter, g.values, color=c, alpha=0.5, s=30, zorder=3)
    ax5c.set_xticks([1, 2]); ax5c.set_xticklabels([LABEL_CASO[1], LABEL_CASO[2]], fontsize=8 * ESCALA_FUENTE)
    ax5c.set_ylabel("Dosis total rescates (mg)", fontsize=9 * ESCALA_FUENTE)
    ax5c.set_title(f"5C - Dosis Total Rescates\nMW p={fmt_p(pv)}", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
    stats_table["Dosis rescates (MW)"] = {"test": "Mann-Whitney U",
                                           "stat": round(u, 2) if not pd.isna(u) else np.nan,
                                           "p": round(pv, 4) if not pd.isna(pv) else np.nan,
                                           "sig": sig_label(pv)}

    ax5d = fig.add_subplot(gs[1, 1])
    e2_plot = e2.merge(df[["Nhc", "Tipo de paciente"]], on="Nhc", how="left")
    for t, c, lbl in [(1, C1, LABEL_CASO[1]), (2, C2, LABEL_CASO[2])]:
        grp_e2 = e2_plot[e2_plot["Tipo de paciente"] == t]
        medianas_x, medianas_y = [], []
        for nhc, pac in grp_e2.groupby("Nhc"):
            pac = pac.sort_values("ts_visita")
            xs  = np.arange(len(pac))
            ys  = pac["dosis_mid_visita"].values
            if len(xs) > 1 and not np.all(np.isnan(ys)):
                ax5d.plot(xs, ys, color=c, alpha=0.18, lw=1)
        max_vis = grp_e2.groupby("Nhc").size().max() if len(grp_e2) > 0 else 0
        if max_vis > 0:
            for pos in range(max_vis):
                vals = []
                for nhc, pac in grp_e2.groupby("Nhc"):
                    pac_s = pac.sort_values("ts_visita")
                    if pos < len(pac_s):
                        v = pac_s["dosis_mid_visita"].iloc[pos]
                        if not pd.isna(v): vals.append(v)
                medianas_x.append(pos)
                medianas_y.append(np.median(vals) if vals else np.nan)
            ax5d.plot(medianas_x, medianas_y, color=c, lw=2.5,
                      label=f"{lbl} (mediana)", zorder=4)
    ax5d.set_xlabel("Visita de seguimiento (orden)", fontsize=9 * ESCALA_FUENTE)
    ax5d.set_ylabel("Dosis Midazolam en infusor (mg)", fontsize=9 * ESCALA_FUENTE)
    ax5d.set_title("5D - Evolucion Temporal de Dosis", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
    ax5d.legend(fontsize=8 * ESCALA_FUENTE)

    fig.savefig(out("P4_Fig5_Gestion_Clinica_Rescates.png"), dpi=180,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig5 guardada")
    return stats_table


# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 6 — SUBGRUPOS POR RANGO DE DOSIS
# ──────────────────────────────────────────────────────────────────────────────

def fig6_subgrupos_dosis(df):
    print("Generando Fig6 — Subgrupos por Rango de Dosis...")
    valid = df[["dosis_acum_mid", "Nhc", "Tipo de paciente",
                "ramsay_ultima", "tiempo_hasta_ram6_h", "uso_rescate"]].dropna(subset=["dosis_acum_mid"]).copy()
    if len(valid) < 9:
        _fig_vacia("P4_Fig6_Subgrupos_Dosis.png", "Fig6: datos insuficientes (n<9)")
        return {}

    n_unique   = valid["dosis_acum_mid"].nunique()
    usar_fijos = False

    if n_unique < 6:
        usar_fijos = True
        print(f"  [WARN] Solo {n_unique} valores unicos de dosis acumulada — usando rangos clinicos fijos")
    else:
        try:
            valid["grupo_dosis"] = pd.qcut(valid["dosis_acum_mid"], q=3,
                                            labels=["Baja dosis", "Dosis media", "Alta dosis"],
                                            duplicates="drop")
            if valid["grupo_dosis"].nunique() < 3:
                usar_fijos = True
                print("  [WARN] qcut genero menos de 3 grupos — usando rangos clinicos fijos")
        except Exception:
            usar_fijos = True
            print("  [WARN] qcut fallo — usando rangos clinicos fijos")

    if usar_fijos:
        bins_fijos   = [-np.inf, 40, 100, np.inf]
        labels_orden = ["Baja (<40 mg)", "Media (40-100 mg)", "Alta (>100 mg)"]
        valid["grupo_dosis"] = pd.cut(valid["dosis_acum_mid"], bins=bins_fijos,
                                       labels=labels_orden)
        titulo_grupos = "Grupos por rangos clinicos (no terciles)"
    else:
        labels_orden  = ["Baja dosis", "Dosis media", "Alta dosis"]
        titulo_grupos = "Subgrupos por Terciles de Dosis Acumulada"

    for g in list(labels_orden):
        n_g = (valid["grupo_dosis"] == g).sum()
        if 0 < n_g < 3:
            idx_adj = labels_orden.index(g)
            target  = labels_orden[idx_adj + 1] if idx_adj < len(labels_orden) - 1 else labels_orden[idx_adj - 1]
            valid.loc[valid["grupo_dosis"] == g, "grupo_dosis"] = target
            print(f"  [WARN] Grupo '{g}' (n={n_g}) fusionado con '{target}'")

    grupos_finales = [g for g in labels_orden if (valid["grupo_dosis"] == g).sum() > 0]
    if len(grupos_finales) < 2:
        _fig_vacia("P4_Fig6_Subgrupos_Dosis.png", "Fig6: grupos insuficientes tras fusion")
        return {}

    stats_table = {}
    for col, lbl in [("ramsay_ultima", "Ramsay final"), ("tiempo_hasta_ram6_h", "Tiempo hasta Ram=6")]:
        grupos_kw = [valid[valid["grupo_dosis"] == t][col].dropna().values for t in grupos_finales]
        grupos_kw = [g for g in grupos_kw if len(g) >= 2]
        if len(grupos_kw) >= 2:
            try:
                h, pv = kruskal(*grupos_kw)
                stats_table[f"Kruskal {lbl}"] = {
                    "test": "Kruskal-Wallis", "stat": round(h, 3),
                    "p": round(pv, 4), "sig": sig_label(pv),
                }
            except Exception:
                pass

    metricas = {
        "% Ramsay = 6":           lambda g: prop_ic95((g["ramsay_ultima"] == 6).sum(), len(g))[0],
        "Tiempo hasta Ram=6 (h)": lambda g: g["tiempo_hasta_ram6_h"].median(),
        "% con rescate":           lambda g: prop_ic95(
            (g["uso_rescate"] > 0).sum(), g["uso_rescate"].notna().sum())[0],
    }
    matriz = {m: [f(valid[valid["grupo_dosis"] == t]) for t in grupos_finales]
              for m, f in metricas.items()}

    colors_g = ["#AED6F1", "#2980B9", "#1A5276"][:len(grupos_finales)]
    x   = np.arange(len(metricas))
    w   = 0.8 / len(grupos_finales)
    off = (len(grupos_finales) - 1) * w / 2

    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.suptitle(f"FIGURA 6 - Analisis de Subgrupos por Dosis [CI=1] — {titulo_grupos}",
                 fontsize=11 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)

    for i, (t, c) in enumerate(zip(grupos_finales, colors_g)):
        vals      = [matriz[m][i] for m in metricas]
        vals_plot = [v if not pd.isna(v) else 0 for v in vals]
        sub  = valid[valid["grupo_dosis"] == t]
        n1   = (sub["Tipo de paciente"] == 1).sum()
        n2   = (sub["Tipo de paciente"] == 2).sum()
        dmin = sub["dosis_acum_mid"].min(); dmax = sub["dosis_acum_mid"].max()
        lbl_bar = f"{t}\nn={len(sub)} (O:{n1}/NO:{n2})\n{dmin:.0f}-{dmax:.0f} mg"
        bars = ax.bar(x + i * w - off, vals_plot, w * 0.9,
                      label=lbl_bar, color=c, alpha=0.85, edgecolor="white")
        for b, v in zip(bars, vals):
            lbl_v = "—" if pd.isna(v) else f"{v:.1f}"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                    lbl_v, ha="center", va="bottom", fontsize=7.5 * ESCALA_FUENTE)

    ax.set_xticks(x); ax.set_xticklabels(list(metricas.keys()), fontsize=9 * ESCALA_FUENTE)
    ax.set_ylabel("Valor (% o horas)", fontsize=9 * ESCALA_FUENTE)
    ax.legend(fontsize=7.5 * ESCALA_FUENTE, title="Grupo dosis", loc="upper right")
    kw_str = " | ".join(f"{k}: p={fmt_p(v['p'])}" for k, v in stats_table.items())
    ax.set_title(f"Kruskal-Wallis — {kw_str}" if kw_str else "Kruskal-Wallis no aplicable",
                 fontsize=8 * ESCALA_FUENTE)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out("P4_Fig6_Subgrupos_Dosis.png"), dpi=180,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig6 guardada")
    return stats_table


# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 7 — TABLA RESUMEN EJECUTIVA
# ──────────────────────────────────────────────────────────────────────────────

def fig7_tabla_resumen(df):
    print("Generando Fig7 — Tabla Resumen Ejecutiva...")
    grupos = {
        "Global":         df,
        "Oncologico":     df[df["Tipo de paciente"] == 1],
        "No Oncologico":  df[df["Tipo de paciente"] == 2],
    }

    def _pct(serie, cond):
        s = serie.dropna(); n = int(cond(s).sum()); N = len(s)
        if N == 0: return "—"
        p, lo, hi = prop_ic95(n, N)
        return f"{p:.1f}% ({n}/{N})\nIC95%[{lo:.1f}-{hi:.1f}]"

    rows = []
    for grp_name, grp in grupos.items():
        dr = grp["dosis_resc_total_e1"].copy()
        if dr.isna().all():
            dr = grp.get("dosis_resc_e2", pd.Series(dtype=float))
        rows.append([
            grp_name,
            _pct(grp["ramsay_ultima"],        lambda s: s == 6),
            med_iqr(grp["dosis_primera_ram6"]),
            med_iqr(grp["tiempo_hasta_ram6_h"]),
            med_iqr(grp["tiempo_fallec_dias"]),
            _pct(grp["uso_rescate"],           lambda s: s > 0),
            med_iqr(dr),
        ])

    cols = ["Grupo", "% Ramsay = 6", "Dosis min. eficaz (mg)\nMed [IQR]",
            "Tiempo hasta sed.\nprofunda (h) Med [IQR]",
            "Tiempo hasta exitus (dias)\nMed [IQR]",
            "% con rescates", "Dosis total rescates (mg)\nMed [IQR]"]

    csv_rows = [{c.replace("\n", " "): r[i] for i, c in enumerate(cols)} for r in rows]
    pd.DataFrame(csv_rows).to_csv(out("P4_tabla_resumen_ejecutiva.csv"),
                                   index=False, encoding="utf-8-sig")

    n_rows  = len(rows) + 1
    fig_h   = max(8.5, n_rows * 2.4 + 3.0)
    # Con la fuente de celda mas grande, 7 columnas ya no caben en 16in de
    # ancho (los encabezados largos se desbordaban entre columnas); se
    # amplia el ancho de figura en vez de reducir la fuente.
    fig, ax = plt.subplots(figsize=(32, fig_h))
    ax.axis("off")
    fig.suptitle("FIGURA 7 - Tabla Resumen Ejecutiva — Analisis Dosis-Eficacia [CI=1]",
                 fontsize=11 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)

    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center",
                   bbox=[0.0, 0.05, 1.0, 0.88])
    tbl.auto_set_font_size(False)

    row_bg = {0: CGLOBAL, 1: "#D6EAF8", 2: "#E8F8F5", 3: "#FEF9E7"}
    for (row_i, col_j), cell in tbl.get_celld().items():
        if row_i == 0:
            cell.set_facecolor(CGLOBAL)
            cell.set_text_props(color="white", fontweight="bold", fontsize=15 * ESCALA_FUENTE)
        else:
            cell.set_facecolor(row_bg.get(row_i, "white"))
            fs = (15 if col_j == 0 else 14.5) * ESCALA_FUENTE
            fw = "bold" if col_j == 0 else "normal"
            cell.set_text_props(fontsize=fs, fontweight=fw)
        cell.set_edgecolor("#AAAAAA" if (row_i > 0 and col_j == 0) else "#CCCCCC")

    # auto_set_column_width se llama despues de fijar el tamano de fuente
    # de cada celda (y tras forzar un draw para refrescar las metricas de
    # texto): si se llama antes, como en el orden original, calcula anchos
    # basandose en el tamano de fuente por defecto de la tabla y las
    # columnas quedan demasiado estrechas para la fuente mas grande.
    fig.canvas.draw()
    tbl.auto_set_column_width(col=list(range(len(cols))))

    fig.tight_layout()
    fig.savefig(out("P4_Fig7_Tabla_Resumen.png"), dpi=180,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig7 guardada | CSV: P4_tabla_resumen_ejecutiva.csv")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 8 — TABLA DE INFERENCIA ESTADÍSTICA
# ──────────────────────────────────────────────────────────────────────────────

def fig8_tabla_inferencia(stats_all):
    print("Generando Fig8 — Tabla de Inferencia Estadistica...")
    rows = []
    for variable, info in stats_all.items():
        test     = info.get("test", "—")
        stat_raw = info.get("stat", np.nan)
        stat_str = _fmt_stat(test, stat_raw)
        p_raw    = info.get("p", np.nan)
        rows.append([
            variable,
            test,
            stat_str,
            str(round(p_raw, 4)) if not pd.isna(p_raw) else "—",
            "Si" if info.get("sig", "ns") != "ns" else "No",
            info.get("sig", "ns"),
        ])

    cols = ["Variable", "Test", "Estadistico", "p-valor", "p < 0.05", "Sig."]
    pd.DataFrame(rows, columns=cols).to_csv(out("P4_tabla_inferencia.csv"),
                                             index=False, encoding="utf-8-sig")

    n_rows  = len(rows) + 1
    fig_h   = max(5.0, n_rows * 0.72 + 2.2)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    fig.suptitle("FIGURA 8 - Tabla de Inferencia Estadistica — PASO 4 [CI=1]",
                 fontsize=11 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)

    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center",
                   bbox=[0.0, 0.05, 1.0, 0.88])
    tbl.auto_set_font_size(False)
    tbl.auto_set_column_width(col=list(range(len(cols))))

    for (row_i, col_j), cell in tbl.get_celld().items():
        if row_i == 0:
            cell.set_facecolor(CGLOBAL)
            cell.set_text_props(color="white", fontweight="bold", fontsize=9 * ESCALA_FUENTE)
        else:
            cell.set_text_props(fontsize=8.5 * ESCALA_FUENTE)
            if col_j == 5 and row_i > 0:
                v = rows[row_i - 1][5]
                cell.set_facecolor("#FADBD8" if v in ("*", "**") else "#D5F5E3")
        cell.set_edgecolor("#CCCCCC")

    fig.tight_layout()
    fig.savefig(out("P4_Fig8_Tabla_Inferencia.png"), dpi=180,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig8 guardada | CSV: P4_tabla_inferencia.csv")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 9 — TIEMPOS CLÍNICOS AVANZADOS (supervivencia y distribución)
# ──────────────────────────────────────────────────────────────────────────────

def fig9_tiempos_avanzados(df):
    print("Generando Fig9 — Tiempos Clinicos Avanzados...")

    df = df.copy()
    df["tiempo_fallec_h"] = df["tiempo_fallec_dias"] * 24

    fig, axes = plt.subplots(2, 2, figsize=(13, 15))
    fig.suptitle(
        "FIGURA 9 — Tiempos Clinicos de la Sedacion Paliativa\n"
        "Caso 1 (Oncologico) vs Caso 2 (No Oncologico)",
        fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL,
    )
    stats_table = {}

    # ── A: Boxplot tiempo sedación → éxitus (horas) ───────────────────────────
    ax = axes[0, 0]
    g1 = df[df["Tipo de paciente"] == 1]["tiempo_fallec_h"].dropna()
    g2 = df[df["Tipo de paciente"] == 2]["tiempo_fallec_h"].dropna()
    u, pv = mw_test(g1, g2)

    data_bp   = [g.values for g in (g1, g2) if len(g) >= 1]
    positions = [i + 1 for i, g in enumerate((g1, g2)) if len(g) >= 1]
    cols_bp   = [c for c, g in zip((C1, C2), (g1, g2)) if len(g) >= 1]

    bp = ax.boxplot(data_bp, positions=positions, patch_artist=True, widths=0.4,
                    medianprops=dict(color="white", lw=2.5),
                    whiskerprops=dict(color=CGLOBAL), capprops=dict(color=CGLOBAL),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5, markerfacecolor="none"))
    for patch, color in zip(bp["boxes"], cols_bp):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    for i, (g, c) in enumerate([(g1, C1), (g2, C2)], 1):
        if len(g) < 1: continue
        jitter = np.random.uniform(-0.12, 0.12, size=len(g))
        ax.scatter(np.full(len(g), i) + jitter, g.values, color=c, alpha=0.55, s=35, zorder=3)
        ax.text(i, float(g.max()) * 1.06 + 0.5,
                f"Med={g.median():.0f}h", ha="center", fontsize=8 * ESCALA_FUENTE, color=c, fontweight="bold")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([LABEL_CASO[1], LABEL_CASO[2]], fontsize=8 * ESCALA_FUENTE)
    ax.set_ylabel("Horas", fontsize=9 * ESCALA_FUENTE)
    ax.set_title(f"A · Tiempo sedacion → exitus (horas)\np={fmt_p(pv)}", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
    stats_table["Fig9A tiempo exitus h (MW)"] = {
        "test": "Mann-Whitney U",
        "stat": round(u, 2) if not pd.isna(u) else np.nan,
        "p": round(pv, 4) if not pd.isna(pv) else np.nan,
        "sig": sig_label(pv),
    }

    # ── B: Histograma distribución tiempo hasta éxitus ────────────────────────
    ax = axes[0, 1]
    all_h = df["tiempo_fallec_h"].dropna()
    if len(all_h) > 0:
        cap   = min(all_h.quantile(0.95) * 1.15, all_h.max() + 5)
        bins  = np.linspace(0, cap, 20)
        for t, c, lbl in [(1, C1, LABEL_CASO[1]), (2, C2, LABEL_CASO[2])]:
            gh = df[df["Tipo de paciente"] == t]["tiempo_fallec_h"].dropna()
            if len(gh) > 0:
                ax.hist(gh, bins=bins, color=c, alpha=0.6, label=lbl, edgecolor="white", lw=0.5)
                ax.axvline(gh.median(), color=c, ls="--", lw=1.5, alpha=0.85)
    ax.set_xlabel("Horas desde inicio de sedacion", fontsize=9 * ESCALA_FUENTE)
    ax.set_ylabel("N pacientes", fontsize=9 * ESCALA_FUENTE)
    ax.set_title("B · Distribucion tiempo hasta exitus", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
    ax.legend(fontsize=8 * ESCALA_FUENTE)
    ax.text(0.02, 0.97, "Linea punteada = mediana", transform=ax.transAxes,
            fontsize=7 * ESCALA_FUENTE, va="top", color="gray")

    # ── C: Curva de supervivencia (Kaplan-Meier simplificada) ─────────────────
    ax = axes[1, 0]
    for t, c, lbl in [(1, C1, LABEL_CASO[1]), (2, C2, LABEL_CASO[2])]:
        times = df[df["Tipo de paciente"] == t]["tiempo_fallec_h"].dropna().sort_values().values
        if len(times) < 2:
            continue
        n      = len(times)
        surv   = np.array([(n - i - 1) / n for i in range(n)])
        t_plot = np.concatenate([[0], times])
        s_plot = np.concatenate([[1.0], surv])
        ax.step(t_plot, s_plot, where="post", color=c, lw=2, label=f"{lbl} (n={n})")
        ax.axvline(np.median(times), color=c, ls=":", lw=1.2, alpha=0.7)
    ax.set_xlabel("Horas desde inicio de sedacion", fontsize=9 * ESCALA_FUENTE)
    ax.set_ylabel("Proporcion supervivientes", fontsize=9 * ESCALA_FUENTE)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5, label="50%")
    ax.set_title("C · Funcion de supervivencia\n(sedacion → exitus)", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
    ax.legend(fontsize=8 * ESCALA_FUENTE)
    ax.text(0.02, 0.05, "Linea punteada vertical = mediana", transform=ax.transAxes,
            fontsize=7 * ESCALA_FUENTE, color="gray")

    # ── D: Tiempo hasta éxitus por categoría de dosis ─────────────────────────
    ax = axes[1, 1]
    dosis_val = df["dosis_primera_ram6"].dropna()
    if len(dosis_val) >= 6:
        med_d = dosis_val.median()
        df_d  = df.dropna(subset=["dosis_primera_ram6", "tiempo_fallec_h"]).copy()
        df_d["grupo_bin"] = df_d["dosis_primera_ram6"].apply(
            lambda x: f"Baja (<=\n{med_d:.0f}mg)" if x <= med_d else f"Alta (>\n{med_d:.0f}mg)")
        grupos = sorted(df_d["grupo_bin"].unique())
        xs     = np.arange(len(grupos))
        for xi, gname in enumerate(grupos):
            for ti, (t, c, lbl) in enumerate([(1, C1, LABEL_CASO[1]), (2, C2, LABEL_CASO[2])]):
                sub_t = df_d[(df_d["grupo_bin"] == gname) &
                             (df_d["Tipo de paciente"] == t)]["tiempo_fallec_h"].dropna()
                off  = (ti - 0.5) * 0.28
                h    = sub_t.median() if len(sub_t) > 0 else 0
                ax.bar(xi + off, h, width=0.25, color=c, alpha=0.82, edgecolor="white",
                       label=lbl if xi == 0 else "")
                if len(sub_t) > 0:
                    ax.text(xi + off, h + 0.5, f"{h:.0f}h",
                            ha="center", fontsize=7.5 * ESCALA_FUENTE, color=c, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(grupos, fontsize=8 * ESCALA_FUENTE)
        ax.set_ylabel("Mediana horas", fontsize=9 * ESCALA_FUENTE)
        ax.set_title("D · Tiempo hasta exitus\npor categoria de dosis", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
        # Techo con margen amplio para que la leyenda superior no tape las
        # etiquetas "Xh" de las barras mas altas.
        ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
        ax.legend(fontsize=8 * ESCALA_FUENTE, loc="upper right",
                  bbox_to_anchor=(1.0, 1.0), framealpha=0.95)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Datos insuficientes para panel D",
                ha="center", va="center", transform=ax.transAxes, fontsize=9 * ESCALA_FUENTE, color="gray")

    fig.tight_layout(h_pad=3.0, w_pad=2.5)
    fig.savefig(out("P4_Fig9_Tiempos_Avanzados.png"), dpi=180,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig9 guardada")
    return stats_table


# ──────────────────────────────────────────────────────────────────────────────
# FIGURA 10 — ANÁLISIS DOSIS ALTA VS BAJA
# ──────────────────────────────────────────────────────────────────────────────

def fig10_dosis_alta_baja(df):
    print("Generando Fig10 — Analisis por Grupos de Dosis (Terciles)...")

    col_dosis = "dosis_acum_mid"
    col_lbl   = "Dosis acumulada Midazolam (mg)"

    if df[col_dosis].notna().sum() < 9:
        _fig_vacia("P4_Fig10_Dosis_Alta_Baja.png", "Fig10: datos insuficientes (n<9)")
        return {}

    df_all = df.copy()
    df_all["alcanzo_r6"] = df_all["ts_primera_ram6"].notna()

    df_g = df_all.dropna(subset=[col_dosis]).copy()
    n_unique = df_g[col_dosis].nunique()
    LABELS_3 = ["Baja dosis", "Dosis media", "Alta dosis"]
    usar_fijos = False

    if n_unique >= 6:
        try:
            df_g["grupo_ter"] = pd.qcut(df_g[col_dosis], q=3,
                                        labels=LABELS_3, duplicates="drop")
            if df_g["grupo_ter"].nunique() < 3:
                usar_fijos = True
        except Exception:
            usar_fijos = True
    else:
        usar_fijos = True

    if usar_fijos:
        bins_fijos = [-np.inf, 40, 100, np.inf]
        df_g["grupo_ter"] = pd.cut(df_g[col_dosis], bins=bins_fijos,
                                   labels=["Baja (<40 mg)", "Media (40-100 mg)", "Alta (>100 mg)"])
        LABELS_3 = ["Baja (<40 mg)", "Media (40-100 mg)", "Alta (>100 mg)"]

    # etiquetas con rangos reales para el eje X
    labels_x = []
    for lbl in LABELS_3:
        sub = df_g[df_g["grupo_ter"] == lbl][col_dosis]
        if len(sub):
            labels_x.append(f"{lbl}\n({sub.min():.0f}-{sub.max():.0f} mg)")
        else:
            labels_x.append(lbl)

    q1, q2 = df_g[col_dosis].quantile([1/3, 2/3]).values
    fig, axes = plt.subplots(2, 2, figsize=(13, 15))
    fig.suptitle(
        "FIGURA 10 — Analisis por Grupos de Dosis Acumulada (Terciles)\n"
        "Baja / Media / Alta dosis   ·   Caso 1 (Oncologico) vs Caso 2 (No Oncologico)",
        fontsize=11 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL, y=0.99,
    )
    stats_table = {}

    # ── A: % que alcanza Ramsay=6 por grupo de dosis y tipo ──────────────────
    ax = axes[0, 0]
    xs    = np.arange(3)
    bar_w = 0.28
    for ti, (t, c, lbl) in enumerate([(1, C1, "Caso 1"), (2, C2, "Caso 2")]):
        pcts = []
        for gd in LABELS_3:
            sub   = df_g[(df_g["grupo_ter"] == gd) & (df_g["Tipo de paciente"] == t)]
            n_tot = len(sub)
            n_r6  = int(sub["alcanzo_r6"].sum()) if n_tot > 0 else 0
            pcts.append((n_r6 / n_tot * 100 if n_tot > 0 else 0, n_r6, n_tot))
        off  = (ti - 0.5) * bar_w
        bars = ax.bar(xs + off, [p[0] for p in pcts], bar_w,
                      color=c, alpha=0.82, label=lbl, edgecolor="white")
        for b, p in zip(bars, pcts):
            ax.text(b.get_x() + b.get_width() / 2, p[0] + 1.5,
                    f"{p[0]:.0f}%\n({p[1]}/{p[2]})", ha="center", fontsize=7 * ESCALA_FUENTE, color=c)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels_x, fontsize=8 * ESCALA_FUENTE)
    ax.set_ylabel("% pacientes que alcanzaron Ramsay = 6", fontsize=9 * ESCALA_FUENTE)
    ax.set_ylim(0, 125)
    ax.legend(fontsize=8 * ESCALA_FUENTE)
    ax.set_title("A · % que alcanza Ramsay=6 por grupo de dosis\ny tipo de paciente",
                 fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    # ── B: Scatter dosis acumulada vs Ramsay máximo ───────────────────────────
    ax = axes[0, 1]
    valid_sc = df_all[[col_dosis, "ramsay_ultima", "Tipo de paciente"]].dropna()
    for t, c, lbl in [(1, C1, LABEL_CASO[1]), (2, C2, LABEL_CASO[2])]:
        sub = valid_sc[valid_sc["Tipo de paciente"] == t]
        ax.scatter(sub[col_dosis], sub["ramsay_ultima"],
                   color=c, label=lbl, alpha=0.72, s=65, edgecolors="white", lw=0.5, zorder=3)
    ax.axvline(q1, color="#95A5A6", ls=":", lw=1.2, alpha=0.8, label=f"P33 ({q1:.0f} mg)")
    ax.axvline(q2, color="#7F8C8D", ls="--", lw=1.2, alpha=0.8, label=f"P67 ({q2:.0f} mg)")
    ax.axhline(6, color="#2ECC71", ls="--", lw=1.2, alpha=0.7, label="Objetivo R=6")
    ax.set_xlabel(col_lbl, fontsize=9 * ESCALA_FUENTE)
    ax.set_ylabel("Ramsay maximo alcanzado", fontsize=9 * ESCALA_FUENTE)
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.legend(fontsize=7.5 * ESCALA_FUENTE)
    ax.set_title("B · Dosis acumulada vs Ramsay maximo\n(todos los pacientes)",
                 fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    # ── C: Dosis acumulada según si se alcanzó Ramsay=6 ──────────────────────
    ax   = axes[1, 0]
    g_no = df_all[~df_all["alcanzo_r6"]][col_dosis].dropna()
    g_si = df_all[ df_all["alcanzo_r6"]][col_dosis].dropna()
    CNARANJA = "#E67E22"
    CVERDE   = "#27AE60"
    for i, (g, c) in enumerate([(g_no, CNARANJA), (g_si, CVERDE)], 1):
        if len(g) < 1: continue
        bp = ax.boxplot(g.values, positions=[i], patch_artist=True, widths=0.4,
                        medianprops=dict(color="white", lw=2),
                        whiskerprops=dict(color=CGLOBAL), capprops=dict(color=CGLOBAL),
                        flierprops=dict(marker="o", markersize=4, alpha=0.4))
        bp["boxes"][0].set_facecolor(c); bp["boxes"][0].set_alpha(0.82)
        jitter = np.random.uniform(-0.13, 0.13, size=len(g))
        ax.scatter(np.full(len(g), i) + jitter, g.values, color=c, alpha=0.55, s=35, zorder=3)
        ax.text(i, float(g.max()) * 1.05 + 0.5, va="bottom",
                s=f"Med={g.median():.0f}", ha="center", fontsize=7.5 * ESCALA_FUENTE, color=c, fontweight="bold")
    # Techo con margen para que la etiqueta de mediana no choque con el
    # titulo de dos lineas del panel (mismo problema que en Regla 7).
    max_c = max([float(g.max()) for g in (g_no, g_si) if len(g) >= 1], default=1)
    ax.set_ylim(top=max_c * 1.35)
    u, pv = mw_test(g_no, g_si)
    if len(g_no) > 1 and len(g_si) > 1:
        sd_pool = np.sqrt((g_si.std() ** 2 + g_no.std() ** 2) / 2)
        d_cohen = (g_si.mean() - g_no.mean()) / sd_pool if sd_pool > 0 else np.nan
    else:
        d_cohen = np.nan
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"No alcanzaron\nRamsay=6\n(n={len(g_no)})",
                        f"Si alcanzaron\nRamsay=6\n(n={len(g_si)})"], fontsize=8 * ESCALA_FUENTE)
    ax.set_ylabel(col_lbl, fontsize=9 * ESCALA_FUENTE)
    d_s = f"{d_cohen:.2f}" if not pd.isna(d_cohen) else "n/c"
    ax.set_title(f"C · Dosis acumulada segun si se alcanzo Ramsay=6\np={fmt_p(pv)}, d Cohen={d_s}",
                 fontsize=9 * ESCALA_FUENTE, fontweight="bold")
    stats_table["Dosis acum por alcance R=6 (MW)"] = {
        "test": "Mann-Whitney U",
        "stat": round(u, 2) if not pd.isna(u) else np.nan,
        "p": round(pv, 4) if not pd.isna(pv) else np.nan,
        "sig": sig_label(pv),
    }

    # ── D: Violin dosis acumulada por tipo, diferenciando si alcanzó R=6 ──────
    ax = axes[1, 1]
    valid_v = df_all[[col_dosis, "Tipo de paciente", "alcanzo_r6"]].dropna(subset=[col_dosis])
    vdata, vpos, vcols = [], [], []
    for t, c in [(1, C1), (2, C2)]:
        g = valid_v[valid_v["Tipo de paciente"] == t][col_dosis].values
        if len(g) >= 3:
            vdata.append(g); vpos.append(t); vcols.append(c)

    if vdata:
        vp = ax.violinplot(vdata, positions=vpos, showmedians=True, showextrema=True)
        for body, c in zip(vp["bodies"], vcols):
            body.set_facecolor(c); body.set_alpha(0.55)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            if part in vp:
                vp[part].set_edgecolor(CGLOBAL)
        for t, c in zip(vpos, vcols):
            sub_t = valid_v[valid_v["Tipo de paciente"] == t]
            for alc, marker, ms in [(True, "o", 40), (False, "X", 50)]:
                pts = sub_t[sub_t["alcanzo_r6"] == alc][col_dosis]
                if len(pts) > 0:
                    jitter = np.random.uniform(-0.12, 0.12, size=len(pts))
                    ax.scatter(np.full(len(pts), t) + jitter, pts.values,
                               color=c if alc else "gray", alpha=0.65, s=ms,
                               marker=marker, zorder=3,
                               label=("Alcanzo R=6" if alc else "No alcanzo R=6") if t == 1 else "")
        ax.axhline(q1, color="#95A5A6", ls=":", lw=1.2, alpha=0.7, label=f"P33 ({q1:.0f} mg)")
        ax.axhline(q2, color="#7F8C8D", ls="--", lw=1.2, alpha=0.7, label=f"P67 ({q2:.0f} mg)")
        ax.legend(fontsize=7.5 * ESCALA_FUENTE, loc="upper right")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([LABEL_CASO[1], LABEL_CASO[2]], fontsize=8 * ESCALA_FUENTE)
    ax.set_ylabel(col_lbl, fontsize=9 * ESCALA_FUENTE)
    ax.set_title("D · Distribucion dosis acumulada por tipo\n(● alcanzo R=6 | ✕ no alcanzo)",
                 fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.92], h_pad=3.0, w_pad=2.5)
    fig.savefig(out("P4_Fig10_Dosis_Alta_Baja.png"), dpi=180,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig10 guardada")
    return stats_table


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: figura vacía
# ──────────────────────────────────────────────────────────────────────────────

def _fig_vacia(filename, mensaje):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, mensaje, ha="center", va="center",
            transform=ax.transAxes, fontsize=12 * ESCALA_FUENTE, color="gray",
            bbox=dict(boxstyle="round", facecolor="#F8F9FA", edgecolor="#CCCCCC"))
    fig.savefig(out(filename), dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[AVISO] {mensaje} — figura vacia guardada: {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    print("\nCargando datos (filtro CI=1)...")
    df, e2 = load_data()

    stats_all = {}

    print("\nGenerando figuras...")
    r = fig1_dosis_respuesta(df);             stats_all.update(r or {})
    r = fig2_proporcion_sedacion(df, e2);     stats_all.update(r or {})
    r = fig3_dosis_minima_eficaz(df);         stats_all.update(r or {})
    r = fig4_tiempos_clinicos(df);            stats_all.update(r or {})
    r = fig5_gestion_rescates(df, e2);        stats_all.update(r or {})
    r = fig6_subgrupos_dosis(df);             stats_all.update(r or {})
    r = fig9_tiempos_avanzados(df);           stats_all.update(r or {})
    r = fig10_dosis_alta_baja(df);            stats_all.update(r or {})
    fig7_tabla_resumen(df)
    fig8_tabla_inferencia(stats_all)

    print("\n" + "=" * 55)
    print("  [OK] PASO 4 COMPLETADO (CI=1)")
    print("  Fig1: Dosis-Respuesta (2 paneles) | Fig2: Sedacion Profunda")
    print("  Fig3: Dosis Min. Eficaz | Fig4: Tiempos Clinicos")
    print("  Fig5: Rescates | Fig6: Subgrupos Dosis")
    print("  Fig9: Tiempos Avanzados + Supervivencia | Fig10: Dosis Alta/Baja")
    print("  Fig7: Tabla Resumen | Fig8: Tabla Inferencia")
    print(f"  Salidas en: {OUTPUT_DIR}")
    if stats_all:
        sig = [k for k, v in stats_all.items() if v.get("sig", "ns") != "ns"]
        print(f"  Resultados significativos (p<0.05): {len(sig)}/{len(stats_all)}")
        for s in sig:
            print(f"    - {s}: p={stats_all[s]['p']}")
    print("=" * 55)
