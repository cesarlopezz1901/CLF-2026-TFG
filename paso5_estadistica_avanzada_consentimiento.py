"""
=============================================================================
PASO 5 - ANALISIS ESTADISTICO AVANZADO
TFG: Manejo del Paciente Paliativo en Sedacion Continua - Hospital La Fe
=============================================================================
Solo pacientes con Consentimiento Informado (CI=1).

BLOQUES:
  Bloque 1 - Analisis de Componentes Principales (ACP)
  Bloque 2 - Inferencia Estadistica
             2A. Regresion logistica (Y = alcanzo Ramsay 6)
             2B. Regresion lineal multiple (Y = dosis minima eficaz)
             2C. Pruebas de hipotesis (Shapiro / t-test / ANOVA / MW / KW)
  Bloque 3 - Analisis de Subgrupos
             3A. Por diagnostico (Oncologico / No oncologico)
             3B. Por rango de dosis (cuartiles)
             3C. Por perfil de complicaciones (con / sin evento adverso)
             3D. Por pauta de administracion (con / sin bolo inicial)
             [Sustituye al subgrupo por dispositivo, ya que TODOS usan infusor]

Fuente de datos: resuelta por utils_io._resolver_paths()
Genera 16 figuras PNG 300 dpi + 5 CSV con resultados estadisticos.
=============================================================================
"""
import os
import sys

# ── Soporte de ejecucion desde subcarpeta del proyecto ────────────────────
# Si el script se ha movido a una subcarpeta (P5_ESTADISTICA_AVANZADA/),
# anadimos la raiz al PYTHONPATH y trabajamos desde alli para que utils_io,
# config_datos.json y los Excel se resuelvan correctamente.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = (os.path.dirname(_SCRIPT_DIR)
         if not os.path.isfile(os.path.join(_SCRIPT_DIR, "utils_io.py"))
         else _SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from utils_io import _resolver_paths, imprimir_configuracion

_E1_PATH, _E2_PATH, _FUENTE, _META = _resolver_paths()
# OUTPUT_DIR resuelto relativo al directorio del script para que las
# salidas vivan junto al script aunque cambiemos el CWD.
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_SCRIPT_DIR, "resultados_paso5_CI"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

imprimir_configuracion("PASO 5 - CONFIGURACION DE DATOS (CI=1)",
                       _E1_PATH, _E2_PATH, _FUENTE, _META)


def out(filename):
    """Devuelve la ruta absoluta del fichero en el directorio de salida."""
    return os.path.join(OUTPUT_DIR, filename)


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

from scipy import stats
from scipy.stats import (shapiro, ttest_ind, f_oneway, mannwhitneyu, kruskal,
                          chi2_contingency, fisher_exact, probplot)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    _HAS_LOWESS = True
except Exception:
    _HAS_LOWESS = False


# =============================================================================
# PALETA Y ESTILO (identica al PASO 3 / PASO 4)
# =============================================================================
C1         = "#C0392B"   # Oncologico
C2         = "#1A5276"   # No oncologico
CGLOBAL    = "#2C3E50"
BG         = "#FAFAFA"
GRID_COLOR = "#E0E0E0"

# Paleta extendida para componentes / cuartiles
PAL_GRAD   = ["#AED6F1", "#5DADE2", "#2980B9", "#1A5276"]
PAL_ALERTA = ["#E67E22", "#D35400"]
PAL_OK     = ["#27AE60", "#229954"]

ESCALA_FUENTE = 1.35  # Regla 1: factor global de escalado de fuente para legibilidad en PDF

plt.rcParams.update({
    "figure.facecolor":  BG,   "axes.facecolor":   BG,
    "axes.edgecolor":    "#CCCCCC", "axes.grid":   True,
    "grid.color":        GRID_COLOR, "grid.linestyle": "--", "grid.alpha": 0.6,
    "font.family":       "DejaVu Sans", "font.size": 9 * ESCALA_FUENTE,
    "axes.titlesize": 10 * ESCALA_FUENTE, "axes.titleweight": "bold",
    "axes.labelsize": 9 * ESCALA_FUENTE,  "xtick.labelsize": 8 * ESCALA_FUENTE, "ytick.labelsize": 8 * ESCALA_FUENTE,
    "legend.fontsize": 8 * ESCALA_FUENTE,  "figure.dpi":       150,
    "savefig.dpi":       300, "savefig.bbox":    "tight",
})

LABEL_CASO = {1: "Caso 1 - Oncologico", 2: "Caso 2 - No Oncologico"}
COLOR_CASO = {1: C1, 2: C2}

# Cohorte (para titulos de figuras)
COHORTE = "Solo CI=1"

# Etiquetas legibles de variables clinicas
LBL_VAR = {
    "edad_num":              "Edad",
    "Edad cuidador":         "Edad del cuidador",
    "Metastasis":            "Metastasis (codigo)",
    "n_sintomas":            "N sintomas refractarios",
    "dosis_inicial_infusor": "Dosis inicial infusor (mg)",
    "dosis_bolo_imp":        "Dosis bolo inicial (mg)",
    "bolo_midazolam":        "Pauta: bolo inicial (0/1)",
    "dosis_acum_mid":        "Dosis acumulada (mg)",
    "ramsay_ultima":         "Ramsay ultima visita",
    "n_rescates_e2":         "N rescates",
    "dosis_primera_ram6":    "Dosis minima eficaz (mg)",
    "tiempo_fallec_dias":    "Tiempo hasta exitus (dias)",
    "tiempo_hasta_ram6_h":   "Tiempo hasta Ramsay=6 (h)",
    "diag_onco":             "Diagnostico oncologico (0/1)",
    "sedacion_6":            "Alcanza Ramsay=6 (0/1)",
    "evento_adverso":        "Evento adverso/complicacion (0/1)",
}

# Predictores fijos (mantener pequenos para n clinico)
PREDICTORES_LOGIT  = ["edad_num", "diag_onco", "dosis_inicial_infusor",
                       "bolo_midazolam", "n_sintomas"]
PREDICTORES_LINEAL = ["edad_num", "diag_onco", "bolo_midazolam", "n_sintomas"]

# Variables numericas para Bloque 2C (pruebas de hipotesis) y PCA
VARS_NUM_HIP = [
    "edad_num", "dosis_inicial_infusor", "dosis_primera_ram6",
    "dosis_acum_mid", "n_sintomas", "n_rescates_e2", "tiempo_fallec_dias",
]
VARS_PCA = [
    "edad_num", "Edad cuidador", "Metastasis", "n_sintomas",
    "dosis_inicial_infusor", "dosis_bolo_imp", "dosis_acum_mid",
    "ramsay_ultima", "n_rescates_e2", "dosis_primera_ram6",
    "tiempo_fallec_dias",
]

# Resumen consolidado (alimentado por cada bloque, consumido por Fig16)
RESUMEN = []


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def _find_col(df, *keywords, require_all=False, exact=False):
    """Localiza columna por substring(s). exact=True hace match exacto."""
    kws_l = [kw.lower() for kw in keywords]
    if exact:
        target = " ".join(kws_l)
        for col in df.columns:
            if col.strip().lower() == target:
                return col
    for col in df.columns:
        col_l = col.strip().lower()
        hits  = [kw in col_l for kw in kws_l]
        if (all(hits) if require_all else any(hits)):
            return col
    return None


def _find_drug_col(df, *keywords):
    """Localiza columna que contenga TODOS los keywords (case-insensitive)."""
    for col in df.columns:
        cl = col.lower()
        if all(kw.lower() in cl for kw in keywords):
            return col
    return None


def _norm_nhc(series):
    """Normaliza IDs: str + strip + lower + elimina .0 de floats."""
    return (
        series.astype(str).str.strip().str.lower()
        .str.replace(r"\.0$", "", regex=True)
    )


def _parse_edad(serie):
    """Convierte 'edad' de texto ('87 anios') a numerico extrayendo digitos."""
    s = serie.astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(s, errors="coerce")


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


def fmt_p(p):
    """Formato compacto de p-valor con marcadores de significacion."""
    if pd.isna(p):  return "—"
    if p < 0.001:   return "<0.001**"
    if p < 0.01:    return f"{p:.3f}**"
    if p < 0.05:    return f"{p:.3f}*"
    return f"{p:.3f} ns"


def sig_label(p):
    if pd.isna(p):  return "ns"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return "ns"


def med_iqr(series):
    s = series.dropna()
    if len(s) == 0: return "—"
    return f"{s.median():.1f} [{s.quantile(.25):.1f}-{s.quantile(.75):.1f}]"


def mean_sd(series):
    s = series.dropna()
    if len(s) == 0: return "—"
    return f"{s.mean():.1f} ± {s.std():.1f}"


def prop_ic95(n, N):
    """Proporcion con IC95% Wilson; devuelve (% pct, lo, hi)."""
    if N == 0: return 0, 0, 0
    p   = n / N
    z   = 1.96
    den = 1 + z**2 / N
    cen = (p + z**2 / (2 * N)) / den
    h   = z * np.sqrt(p * (1 - p) / N + z**2 / (4 * N**2)) / den
    return p * 100, max(0, (cen - h) * 100), min(100, (cen + h) * 100)


def chi2_or_fisher(ct):
    """Chi2 si expected>=5 en todas celdas, sino Fisher (estad. = OR)."""
    ct = np.array(ct)
    if ct.shape == (2, 2):
        exp = chi2_contingency(ct)[3]
        if (exp < 5).any():
            _, p = fisher_exact(ct)
            a, b, c, d = ct[0, 0], ct[0, 1], ct[1, 0], ct[1, 1]
            if b * c > 0:
                or_v = round(float(a * d) / float(b * c), 4)
            elif a * d > 0:
                or_v = np.inf
            else:
                or_v = 0.0
            return "Fisher", or_v, p
    chi2, p, _, _ = chi2_contingency(ct)
    return "Chi2", chi2, p


def _fmt_stat(test_name, stat_val):
    """Formatea estadistico: OR=X.XX para Fisher, numerico para otros tests."""
    if pd.isna(stat_val) or stat_val is None: return "—"
    if test_name == "Fisher":
        if stat_val == np.inf: return "OR=inf"
        if stat_val == 0.0:    return "OR=0"
        return f"OR={float(stat_val):.2f}"
    return f"{float(stat_val):.3f}"


def _auto_test_2groups(g1, g2):
    """Selecciona t-test o Mann-Whitney segun Shapiro. Devuelve (test,stat,p)."""
    g1, g2 = pd.Series(g1).dropna(), pd.Series(g2).dropna()
    if len(g1) < 2 or len(g2) < 2:
        return "—", np.nan, np.nan
    if len(g1) < 3 or len(g2) < 3:
        u, p = mannwhitneyu(g1, g2, alternative="two-sided")
        return "Mann-Whitney U", u, p
    try:
        sh1 = shapiro(g1)[1]
        sh2 = shapiro(g2)[1]
    except Exception:
        sh1, sh2 = 0.0, 0.0
    if sh1 > 0.05 and sh2 > 0.05:
        t, p = ttest_ind(g1, g2, equal_var=False)
        return "t-test (Welch)", t, p
    u, p = mannwhitneyu(g1, g2, alternative="two-sided")
    return "Mann-Whitney U", u, p


def _auto_test_kgroups(groups):
    """ANOVA o Kruskal-Wallis segun normalidad de cada grupo."""
    groups = [pd.Series(g).dropna() for g in groups]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return "—", np.nan, np.nan
    if any(len(g) < 3 for g in groups):
        h, p = kruskal(*groups)
        return "Kruskal-Wallis", h, p
    try:
        norms = [shapiro(g)[1] > 0.05 for g in groups]
    except Exception:
        norms = [False]
    if all(norms):
        f, p = f_oneway(*groups)
        return "ANOVA", f, p
    h, p = kruskal(*groups)
    return "Kruskal-Wallis", h, p


def _annot_pval(ax, x1, x2, y, p, color=None):
    """Dibuja corchete con p-valor entre x1 y x2 a altura y."""
    color = color or CGLOBAL
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c=color)
    ax.text((x1 + x2) / 2, y + h, fmt_p(p), ha="center", va="bottom",
            fontsize=8 * ESCALA_FUENTE, color=color)


def _fig_vacia(filename, mensaje):
    """Figura placeholder cuando n insuficiente."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(0.5, 0.5, mensaje, ha="center", va="center",
            transform=ax.transAxes, fontsize=13 * ESCALA_FUENTE, color="gray",
            bbox=dict(boxstyle="round", facecolor="#F8F9FA",
                       edgecolor="#CCCCCC"))
    fig.savefig(out(filename), facecolor=BG)
    plt.close(fig)
    print(f"  [AVISO] {mensaje} -> figura vacia: {filename}")


def _add_resumen(bloque, analisis, resultado, p, interpretacion):
    """Anade fila al resumen consolidado final (Fig16)."""
    RESUMEN.append({
        "Bloque":         bloque,
        "Analisis":       analisis,
        "Resultado":      resultado,
        "p-valor":        ("—" if pd.isna(p) else
                            ("<0.001" if p < 0.001 else f"{p:.3f}")),
        "Interpretacion": interpretacion,
    })


# =============================================================================
# CARGA Y PREPARACION DE DATOS
# =============================================================================

def load_data():
    """Lee E1+E2, calcula agregados por paciente y devuelve (df, e2)."""
    e1 = pd.read_excel(_E1_PATH)
    e2 = pd.read_excel(_E2_PATH)
    e1.columns = e1.columns.str.strip()
    e2.columns = e2.columns.str.strip()

    # ── columnas clave E1 ────────────────────────────────────────────────
    col_nhc_e1     = _find_col(e1, "nhc") or _find_col(e1, "historia") or _find_col(e1, "codigo")
    col_tipo       = _find_col(e1, "tipo", "paciente", require_all=True) \
                      or _find_col(e1, "tipo")
    col_ci         = _find_col(e1, "consentimiento") or next(
        (c for c in e1.columns if c.strip().lower() == "ci"
         or "consent" in c.lower()), None)
    col_fecha_ini  = _find_col(e1, "inicio", "sedaci", require_all=True) \
                      or _find_col(e1, "inicio")
    col_hora_ini   = _find_col(e1, "hora", "inicio", require_all=True)
    col_fecha_fall = (_find_col(e1, "fallecimiento") or _find_col(e1, "exitus")
                       or _find_col(e1, "muerte"))
    col_hora_fall  = (_find_col(e1, "hora", "xitus", require_all=True)
                       or _find_col(e1, "hora", "fall", require_all=True))
    col_dosis_bolo    = (_find_col(e1, "dosis bolo", exact=True)
                          or _find_col(e1, "dosis", "bolo", require_all=True))
    col_bolo_mid      = _find_col(e1, "bolo", "midazolam", require_all=True)
    col_dosis_inic    = _find_drug_col(e1, "dosis", "midazolam")  # "Dosis de Midazolam en"
    col_rescate_e1    = (_find_col(e1, "utilizacion", "rescate", require_all=True)
                          or _find_col(e1, "rescate"))
    col_dosis_resc_e1 = _find_col(e1, "dosis", "rescate", require_all=True)
    col_edad          = _find_col(e1, "edad", exact=True) or "edad"
    col_edad_cuid     = _find_col(e1, "edad", "cuidador", require_all=True)
    col_metast        = _find_col(e1, "metastasis")
    col_sexo          = _find_col(e1, "sexo", exact=True) or _find_col(e1, "sexo")

    # ── normalizar E1 ────────────────────────────────────────────────────
    if col_nhc_e1 and col_nhc_e1 != "Nhc":
        e1["Nhc"] = e1[col_nhc_e1]
    e1["Nhc"] = _norm_nhc(e1["Nhc"])

    if col_tipo:
        e1["Tipo de paciente"] = pd.to_numeric(e1[col_tipo], errors="coerce")

    # parsing fechas
    if col_fecha_ini:
        e1[col_fecha_ini] = pd.to_datetime(e1[col_fecha_ini],
                                            errors="coerce", dayfirst=True)
    if col_fecha_fall:
        e1[col_fecha_fall] = pd.to_datetime(e1[col_fecha_fall],
                                             errors="coerce", dayfirst=True)
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
    neg = e1["tiempo_fallec_dias"].notna() & (e1["tiempo_fallec_dias"] < 0)
    if neg.any():
        e1.loc[neg, "tiempo_fallec_dias"] = np.nan

    # numerizar columnas de farmacos/rescates
    e1["dosis_bolo_mid_e1"]   = pd.to_numeric(
        e1[col_dosis_bolo] if col_dosis_bolo else pd.Series([np.nan] * len(e1)),
        errors="coerce")
    e1["bolo_midazolam"]      = pd.to_numeric(
        e1[col_bolo_mid] if col_bolo_mid else pd.Series([np.nan] * len(e1)),
        errors="coerce")
    e1["dosis_inicial_infusor"] = pd.to_numeric(
        e1[col_dosis_inic] if col_dosis_inic else pd.Series([np.nan] * len(e1)),
        errors="coerce")
    e1["uso_rescate"]         = pd.to_numeric(
        e1[col_rescate_e1] if col_rescate_e1 else pd.Series([np.nan] * len(e1)),
        errors="coerce")
    e1["dosis_resc_total_e1"] = pd.to_numeric(
        e1[col_dosis_resc_e1] if col_dosis_resc_e1 else pd.Series([np.nan] * len(e1)),
        errors="coerce")

    # demograficas / clinicas adicionales
    e1["edad_num"]      = _parse_edad(e1[col_edad]) if col_edad in e1.columns else np.nan
    e1["Edad cuidador"] = pd.to_numeric(
        e1[col_edad_cuid] if col_edad_cuid else pd.Series([np.nan] * len(e1)),
        errors="coerce")
    e1["Metastasis"]    = pd.to_numeric(
        e1[col_metast] if col_metast else pd.Series([np.nan] * len(e1)),
        errors="coerce")
    if col_sexo and col_sexo in e1.columns:
        e1["sexo_num"] = pd.to_numeric(e1[col_sexo], errors="coerce")
    else:
        e1["sexo_num"] = np.nan

    # n_sintomas refractarios (suma de columnas Sintomas-*)
    sint_cols = [c for c in e1.columns if c.strip().lower().startswith("sintomas-")]
    if sint_cols:
        sint_num = e1[sint_cols].apply(lambda c: pd.to_numeric(c, errors="coerce"))
        e1["n_sintomas"] = sint_num.fillna(0).sum(axis=1)
    else:
        e1["n_sintomas"] = 0

    # diag_onco: 1 si tipo 1 (oncologico), 0 si tipo 2
    e1["diag_onco"] = (e1["Tipo de paciente"] == 1).astype(float)

    # dosis_bolo_imp: dosis bolo imputada con 0 (clinicamente: sin bolo = 0 mg)
    e1["dosis_bolo_imp"] = e1["dosis_bolo_mid_e1"].fillna(0.0)

    # ── columnas clave E2 ────────────────────────────────────────────────
    col_nhc_e2     = _find_col(e2, "nhc") or _find_col(e2, "historia") or _find_col(e2, "codigo")
    col_ramsay     = _find_col(e2, "ramsay")
    col_fvis       = _find_col(e2, "fecha", "visita", require_all=True) \
                      or _find_col(e2, "fecha")
    col_hvis       = _find_col(e2, "hora", "visita", require_all=True) \
                      or _find_col(e2, "hora")
    col_mid_e2     = _find_drug_col(e2, "midazolam", "infusor") \
                      or _find_drug_col(e2, "midazolam")
    col_resc_e2    = _find_col(e2, "rescate")
    col_dosis_re2  = _find_col(e2, "dosis", "rescate", require_all=True)
    col_complic    = _find_col(e2, "complicaciones")
    col_reemplazo  = _find_col(e2, "reemplazo", "infusor", require_all=True) \
                      or _find_col(e2, "reemplazo")
    col_buen_ctrl  = _find_col(e2, "buen control")

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
        e2[col_mid_e2] if col_mid_e2 else pd.Series([np.nan] * len(e2)),
        errors="coerce")
    e2["dosis_resc_visita"] = pd.to_numeric(
        e2[col_dosis_re2] if col_dosis_re2 else
        (e2[col_resc_e2] if col_resc_e2 else pd.Series([np.nan] * len(e2))),
        errors="coerce")
    e2["uso_resc_visita"] = pd.to_numeric(
        e2[col_resc_e2] if col_resc_e2 else pd.Series([np.nan] * len(e2)),
        errors="coerce").gt(0).astype(float)
    e2["complic_v"] = pd.to_numeric(
        e2[col_complic] if col_complic else pd.Series([np.nan] * len(e2)),
        errors="coerce")
    e2["reemplazo_v"] = pd.to_numeric(
        e2[col_reemplazo] if col_reemplazo else pd.Series([np.nan] * len(e2)),
        errors="coerce")
    e2["buen_ctrl_v"] = pd.to_numeric(
        e2[col_buen_ctrl] if col_buen_ctrl else pd.Series([np.nan] * len(e2)),
        errors="coerce")

    # ── validacion merge ─────────────────────────────────────────────────
    nhc_e1  = set(e1["Nhc"].dropna())
    nhc_e2  = set(e2["Nhc"].dropna())
    matched = nhc_e1 & nhc_e2
    print("=== VALIDACION DE MERGE ===")
    print(f"Pacientes E1: {len(nhc_e1)} | Unicos E2: {len(nhc_e2)} | "
          f"Match: {len(matched)} | Sin E2: {len(nhc_e1 - nhc_e2)}")

    # ── agregados E2 por paciente ────────────────────────────────────────
    def _agg_e2(e2_df, e1_nhcs):
        rows = []
        for nhc, grp in e2_df[e2_df["Nhc"].isin(e1_nhcs)].groupby("Nhc"):
            grp = grp.sort_values("ts_visita")

            dosis_acum = grp["dosis_mid_visita"].sum(min_count=1)
            ram_first  = grp["Escala Ramsay"].dropna().iloc[0]  if grp["Escala Ramsay"].notna().any() else np.nan
            ram_last   = grp["Escala Ramsay"].dropna().iloc[-1] if grp["Escala Ramsay"].notna().any() else np.nan

            ram6 = grp[grp["Escala Ramsay"] == 6]
            if len(ram6):
                ts_pram6     = ram6["ts_visita"].iloc[0]
                dosis_pram6  = ram6["dosis_mid_visita"].iloc[0]
                ramsay_pram6 = ram6["Escala Ramsay"].iloc[0]
            else:
                ts_pram6, dosis_pram6, ramsay_pram6 = pd.NaT, np.nan, np.nan

            ram6_m = grp[grp["Escala Ramsay"] == 6]
            mant_h = np.nan
            if len(ram6_m) >= 2:
                d = ram6_m["ts_visita"].diff().dt.total_seconds().dropna() / 3600
                mant_h = float(d.sum())

            n_resc     = int(grp["uso_resc_visita"].fillna(0).gt(0).sum())
            dosis_resc = grp["dosis_resc_visita"].sum(min_count=1)

            # eventos adversos (composite)
            n_compl    = int((grp["complic_v"].fillna(0) > 0).sum())
            n_reempl   = int((grp["reemplazo_v"].fillna(0) > 0).sum())
            n_malctrl  = int((grp["buen_ctrl_v"] == 0).sum())
            ev_complic = int((n_compl + n_reempl) > 0)
            ev_advcomp = int((n_compl + n_reempl + n_malctrl) > 0)

            rows.append({
                "Nhc":                 nhc,
                "tiene_e2":            1,
                "dosis_acum_mid":      float(dosis_acum)   if not pd.isna(dosis_acum)   else np.nan,
                "ramsay_primera":      float(ram_first)    if not pd.isna(ram_first)    else np.nan,
                "ramsay_ultima":       float(ram_last)     if not pd.isna(ram_last)     else np.nan,
                "ts_primera_ram6":     ts_pram6,
                "dosis_primera_ram6":  float(dosis_pram6)  if not pd.isna(dosis_pram6)  else np.nan,
                "ramsay_primera_ram6": float(ramsay_pram6) if not pd.isna(ramsay_pram6) else np.nan,
                "tiempo_mant_h":       mant_h,
                "n_rescates_e2":       n_resc,
                "dosis_resc_e2":       float(dosis_resc) if not pd.isna(dosis_resc) else np.nan,
                "n_complic":           n_compl,
                "n_reempl":            n_reempl,
                "n_malctrl":           n_malctrl,
                "evento_infusor":      ev_complic,
                "evento_adverso":      ev_advcomp,
            })
        return pd.DataFrame(rows)

    agg = _agg_e2(e2, nhc_e1)
    e1  = e1.merge(agg, on="Nhc", how="left")
    e1["tiene_e2"] = e1["tiene_e2"].fillna(0).astype(int)

    e1["tiempo_hasta_ram6_h"] = (
        (e1["ts_primera_ram6"] - e1["ts_inicio_sed"]).dt.total_seconds() / 3600
    )

    # sedacion_6: definido SOLO para pacientes con visitas E2
    sedacion = pd.Series(np.nan, index=e1.index, dtype=float)
    mask_e2  = e1["tiene_e2"] == 1
    sedacion.loc[mask_e2] = e1.loc[mask_e2, "ts_primera_ram6"].notna().astype(float)
    e1["sedacion_6"] = sedacion

    # evento_adverso queda NaN si no hay E2
    e1.loc[~mask_e2, "evento_adverso"] = np.nan
    e1.loc[~mask_e2, "evento_infusor"] = np.nan

    # ── FILTRO CONSENTIMIENTO INFORMADO ──────────────────────────────────
    if col_ci:
        ci_vals = pd.to_numeric(e1[col_ci], errors="coerce")
        n_total = len(e1)
        e1 = e1[ci_vals == 1].reset_index(drop=True)
        print(f"   [CI] {len(e1)} pacientes con CI=1 ({n_total - len(e1)} excluidos)")
        nhc_ci = set(e1["Nhc"].dropna())
        e2 = e2[e2["Nhc"].isin(nhc_ci)].reset_index(drop=True)
        # recomputar mask_e2 tras filtrado
        mask_e2 = e1["tiene_e2"] == 1
    else:
        print("   [CI] AVISO: columna CI no encontrada -> se incluyen todos los pacientes")
    # ─────────────────────────────────────────────────────────────────────

    e1["tipo_label"] = e1["Tipo de paciente"].map(LABEL_CASO)

    print(f"   E1: {len(e1)} pacientes | E2: {len(e2)} visitas | "
          f"con E2: {int(mask_e2.sum())}")
    print(f"   Sedacion=6 alcanzada: {int(e1['sedacion_6'].sum())} / "
          f"{int(e1['sedacion_6'].notna().sum())} (con seguimiento)")
    print(f"   Evento adverso composito: "
          f"{int((e1['evento_adverso'] == 1).sum())} pacientes")
    print(f"   edad_num: media={e1['edad_num'].mean():.1f}, "
          f"n_valid={e1['edad_num'].notna().sum()}")

    return e1, e2


# =============================================================================
# BLOQUE 1 - ANALISIS DE COMPONENTES PRINCIPALES (ACP / PCA)
# =============================================================================

def _prep_pca(df):
    """Prepara matriz X estandarizada para PCA. Devuelve (X_std, var_names,
    df_valido, n_imputado_por_var, mask_valido_idx)."""
    # Selecciona variables con > 50% de cobertura
    vars_ok = []
    n_imp   = {}
    for v in VARS_PCA:
        if v not in df.columns: continue
        nv = df[v].notna().sum()
        if nv / max(len(df), 1) >= 0.5:
            vars_ok.append(v)
            n_imp[v] = int(df[v].isna().sum())
    if len(vars_ok) < 3:
        return None, [], df.iloc[0:0], {}, []

    X = df[vars_ok].copy()
    # Imputacion por mediana (registrada)
    for v in vars_ok:
        if X[v].isna().any():
            X[v] = X[v].fillna(X[v].median())

    # Estandarizar
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    return Xs, vars_ok, df, n_imp, X.index.tolist()


def fig01_acp_varianza(df):
    """Fig01: Scree plot + Varianza acumulada (graficos requeridos 1 y 2)."""
    print("Generando Fig01 - ACP Varianza...")
    Xs, vars_ok, _, n_imp, _ = _prep_pca(df)
    if Xs is None or len(vars_ok) < 3 or len(df) < 4:
        _fig_vacia("P5_Fig01_ACP_Varianza.png",
                    "Fig01: variables insuficientes para PCA")
        return {}

    n_comp = min(len(vars_ok), Xs.shape[0])
    pca = PCA(n_components=n_comp).fit(Xs)
    var_ratio   = pca.explained_variance_ratio_
    var_cum     = np.cumsum(var_ratio)
    n_para_80   = int(np.searchsorted(var_cum, 0.80) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5))
    fig.suptitle(f"FIGURA 1 - ACP: Varianza Explicada — {COHORTE}",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)

    # Panel A: Scree plot
    ax = axes[0]
    xs = np.arange(1, n_comp + 1)
    bars = ax.bar(xs, var_ratio * 100, color=PAL_GRAD[1], alpha=0.8,
                   edgecolor="white", label="Varianza explicada")
    ax.plot(xs, var_ratio * 100, "o-", color=CGLOBAL, lw=2, markersize=6,
             label="Tendencia")
    for b, v in zip(bars, var_ratio):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"{v*100:.1f}%", ha="center", fontsize=8 * ESCALA_FUENTE, color=CGLOBAL)
    ax.set_xticks(xs)
    ax.set_xlabel("Componente principal", fontsize=9 * ESCALA_FUENTE)
    ax.set_ylabel("Varianza explicada (%)", fontsize=9 * ESCALA_FUENTE)
    ax.set_title("A · Scree plot (varianza por componente)",
                 fontsize=10 * ESCALA_FUENTE, fontweight="bold")
    ax.legend(fontsize=8 * ESCALA_FUENTE)

    # Panel B: Varianza acumulada
    ax = axes[1]
    ax.plot(xs, var_cum * 100, "o-", color=C2, lw=2.5, markersize=7,
             label="Varianza acumulada")
    ax.fill_between(xs, 0, var_cum * 100, color=C2, alpha=0.15)
    ax.axhline(80, color="#E74C3C", ls="--", lw=1.4, alpha=0.85,
                label="Umbral 80%")
    ax.axvline(n_para_80, color="#27AE60", ls=":", lw=1.4, alpha=0.85,
                label=f"{n_para_80} componente(s) >= 80%")
    for x, v in zip(xs, var_cum):
        ax.text(x, v * 100 + 2.5, f"{v*100:.0f}%",
                ha="center", fontsize=7.5 * ESCALA_FUENTE, color=CGLOBAL)
    ax.set_xticks(xs)
    ax.set_xlabel("N de componentes", fontsize=9 * ESCALA_FUENTE)
    ax.set_ylabel("Varianza acumulada (%)", fontsize=9 * ESCALA_FUENTE)
    ax.set_ylim(0, 110)
    ax.set_title(f"B · Varianza acumulada (>= 80% con {n_para_80} comp.)",
                 fontsize=10 * ESCALA_FUENTE, fontweight="bold")
    ax.legend(fontsize=8 * ESCALA_FUENTE, loc="lower right")

    # Nota de variables / imputaciones
    nota = (f"n pacientes = {Xs.shape[0]}, n variables = {len(vars_ok)}.  "
            f"Imputacion mediana en: " +
            ", ".join(f"{v} ({n_imp[v]})" for v in vars_ok if n_imp[v] > 0)
            if any(v > 0 for v in n_imp.values()) else
            f"n pacientes = {Xs.shape[0]}, n variables = {len(vars_ok)}. "
            "Sin imputacion.")
    fig.text(0.5, 0.005, nota, ha="center", fontsize=7.5 * ESCALA_FUENTE, color="gray")

    fig.tight_layout(rect=[0, 0.04, 1, 0.88], w_pad=2.8)
    fig.savefig(out("P5_Fig01_ACP_Varianza.png"), facecolor=BG)
    plt.close(fig)
    print("[OK] Fig01 guardada")

    _add_resumen("ACP", "Varianza explicada (>=80%)",
                  f"{n_para_80} componentes ({var_cum[n_para_80-1]*100:.1f}%)",
                  np.nan,
                  "Reduccion eficaz de dimensionalidad con pocos componentes"
                  if n_para_80 <= 3 else
                  "Estructura multidimensional, requiere varios componentes")

    return {"[ACP] Varianza explicada PC1":
              {"test": "PCA", "stat": round(float(var_ratio[0]) * 100, 2),
               "p": np.nan, "sig": "—"},
            "[ACP] N componentes para 80%":
              {"test": "PCA", "stat": float(n_para_80),
               "p": np.nan, "sig": "—"}}


def fig02_acp_biplot(df):
    """Fig02: Biplot PC1 vs PC2 (grafico requerido 3)."""
    print("Generando Fig02 - ACP Biplot...")
    Xs, vars_ok, _, _, idx = _prep_pca(df)
    if Xs is None or len(vars_ok) < 3 or Xs.shape[0] < 4:
        _fig_vacia("P5_Fig02_ACP_Biplot.png",
                    "Fig02: datos insuficientes para biplot")
        return {}

    pca = PCA(n_components=min(2, len(vars_ok))).fit(Xs)
    scores   = pca.transform(Xs)
    loadings = pca.components_.T  # filas=variables, cols=PCs
    vr       = pca.explained_variance_ratio_

    sub_df = df.loc[idx]
    tipos  = sub_df["Tipo de paciente"].values

    fig, ax = plt.subplots(figsize=(9.5, 8))
    fig.suptitle(f"FIGURA 2 - ACP: Biplot PC1 vs PC2 — {COHORTE}",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)

    # Scatter de pacientes
    for t in (1, 2):
        mask = tipos == t
        if mask.sum() == 0: continue
        ax.scatter(scores[mask, 0], scores[mask, 1],
                    c=COLOR_CASO[t], label=LABEL_CASO[t],
                    s=80, alpha=0.75, edgecolors="white", linewidths=0.6,
                    zorder=3)

    # Flechas de loadings (escaladas a la nube de puntos)
    scale = 0.75 * max(np.abs(scores).max(), 1e-9) / max(np.abs(loadings).max(),
                                                          1e-9)
    texts = []
    for j, v in enumerate(vars_ok):
        ax.arrow(0, 0, loadings[j, 0] * scale, loadings[j, 1] * scale,
                 head_width=0.12, length_includes_head=True,
                 color="#7B7D7D", alpha=0.85, zorder=4)
        t = ax.text(loadings[j, 0] * scale * 1.15, loadings[j, 1] * scale * 1.15,
                    LBL_VAR.get(v, v), fontsize=8 * ESCALA_FUENTE, color="#34495E",
                    ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                               edgecolor="#BDC3C7", alpha=0.9))
        texts.append(t)

    from adjustText import adjust_text
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    ax.axhline(0, color="gray", lw=0.7, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.7, alpha=0.5)
    ax.set_xlabel(f"PC1 ({vr[0]*100:.1f}% varianza)", fontsize=10 * ESCALA_FUENTE)
    ax.set_ylabel(f"PC2 ({vr[1]*100:.1f}% varianza)" if len(vr) > 1
                   else "PC2 (n/a)", fontsize=10 * ESCALA_FUENTE)
    ax.set_title("Pacientes (puntos) y contribucion de variables (flechas)",
                  fontsize=10 * ESCALA_FUENTE, fontweight="bold")
    ax.legend(fontsize=9 * ESCALA_FUENTE, loc="best", framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out("P5_Fig02_ACP_Biplot.png"), facecolor=BG)
    plt.close(fig)
    print("[OK] Fig02 guardada")
    return {}


def fig03_acp_loadings(df):
    """Fig03: Heatmap de loadings (grafico requerido 4)."""
    print("Generando Fig03 - ACP Loadings Heatmap...")
    Xs, vars_ok, _, _, _ = _prep_pca(df)
    if Xs is None or len(vars_ok) < 3:
        _fig_vacia("P5_Fig03_ACP_Loadings.png",
                    "Fig03: datos insuficientes para loadings")
        return {}

    n_comp = min(len(vars_ok), 6, Xs.shape[0])
    pca = PCA(n_components=n_comp).fit(Xs)
    loadings = pca.components_.T  # filas=variables, cols=PCs
    var_ratio = pca.explained_variance_ratio_

    cols = [f"PC{i+1}\n({var_ratio[i]*100:.1f}%)" for i in range(n_comp)]
    rows = [LBL_VAR.get(v, v) for v in vars_ok]
    df_load = pd.DataFrame(loadings, index=rows, columns=cols)

    # Exportar a CSV
    df_load.to_csv(out("P5_acp_loadings.csv"), encoding="utf-8-sig",
                    float_format="%.4f")

    fig, ax = plt.subplots(figsize=(10, max(7, 0.68 * len(rows) + 2.2)))
    fig.suptitle(f"FIGURA 3 - ACP: Heatmap de Loadings — {COHORTE}",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)

    sns.heatmap(df_load, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                 vmin=-1, vmax=1, cbar_kws={"label": "Loading (contribucion)"},
                 linewidths=0.5, linecolor="white", ax=ax,
                 annot_kws={"fontsize": 8 * ESCALA_FUENTE})
    ax.set_title("Contribucion de cada variable a cada componente principal",
                  fontsize=10 * ESCALA_FUENTE, fontweight="bold")
    ax.set_xlabel("Componente", fontsize=9 * ESCALA_FUENTE)
    ax.set_ylabel("Variable clinica", fontsize=9 * ESCALA_FUENTE)
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Interpretacion auto: top-3 variables por |loading| en PC1 y PC2
    inter_txt = []
    for k in range(min(3, n_comp)):
        order = np.argsort(-np.abs(loadings[:, k]))[:3]
        top   = ", ".join(f"{LBL_VAR.get(vars_ok[i], vars_ok[i])} "
                            f"({loadings[i,k]:+.2f})" for i in order)
        inter_txt.append(f"PC{k+1}: {top}")
    fig.text(0.5, 0.005, " | ".join(inter_txt),
              ha="center", fontsize=7.5 * ESCALA_FUENTE, color="gray", wrap=True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out("P5_Fig03_ACP_Loadings.png"), facecolor=BG)
    plt.close(fig)
    print("[OK] Fig03 guardada | CSV: P5_acp_loadings.csv")

    _add_resumen("ACP", "Variable dominante en PC1",
                  LBL_VAR.get(vars_ok[int(np.argmax(np.abs(loadings[:, 0])))],
                              vars_ok[int(np.argmax(np.abs(loadings[:, 0])))]),
                  np.nan,
                  "Identifica eje principal de variacion entre pacientes")
    return {}


# =============================================================================
# BLOQUE 2A - REGRESION LOGISTICA  (Y = sedacion_6)
# =============================================================================

def _fit_logit_inference(X_df, y):
    """Ajusta Logit (statsmodels); fallback a sklearn+bootstrap si falla."""
    res = {}
    Xc  = sm.add_constant(X_df, has_constant="add")
    try:
        m = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
        if not np.all(np.isfinite(m.bse)) or np.any(m.bse > 1e3):
            raise ValueError("bse no finito / coef inestable")
        ci = m.conf_int()
        for v in X_df.columns:
            res[v] = dict(
                coef=float(m.params[v]),
                OR=float(np.exp(m.params[v])),
                ci_lo=float(np.exp(ci.loc[v, 0])),
                ci_hi=float(np.exp(ci.loc[v, 1])),
                p=float(m.pvalues[v]),
            )
        return res, "Logit (max. verosimilitud)", m
    except Exception as e:
        print(f"  [WARN] Logit statsmodels fallo ({e}); fallback sklearn + bootstrap")
        clf = LogisticRegression(max_iter=2000, C=1e6).fit(X_df.values, y)
        coefs = clf.coef_[0]
        rng   = np.random.RandomState(42)
        boot  = []
        Xv    = X_df.values
        for _ in range(300):
            idx = rng.choice(len(y), len(y), replace=True)
            if len(np.unique(y[idx])) < 2:
                continue
            try:
                c = LogisticRegression(max_iter=2000, C=1e6).fit(Xv[idx], y[idx]).coef_[0]
                boot.append(c)
            except Exception:
                pass
        boot = np.array(boot) if boot else np.zeros((1, len(coefs)))
        for j, v in enumerate(X_df.columns):
            if len(boot) > 1:
                lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
            else:
                lo, hi = np.nan, np.nan
            res[v] = dict(coef=float(coefs[j]), OR=float(np.exp(coefs[j])),
                          ci_lo=float(np.exp(lo)) if not pd.isna(lo) else np.nan,
                          ci_hi=float(np.exp(hi)) if not pd.isna(hi) else np.nan,
                          p=np.nan)
        return res, "sklearn + bootstrap (separacion: p no disponible)", clf


def _eval_logistic_pred(X_df, y):
    """Devuelve (y_true_eval, prob_pos_eval, metodo). Decide split o CV
    segun n y balance de clases."""
    n    = len(y)
    npos = int((y == 1).sum())
    nneg = int((y == 0).sum())
    Xv   = X_df.values
    scaler = StandardScaler()
    Xv_s   = scaler.fit_transform(Xv)

    if n >= 60 and nneg >= 20 and npos >= 20:
        Xtr, Xte, ytr, yte = train_test_split(
            Xv_s, y, test_size=0.2, stratify=y, random_state=42)
        clf  = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        return yte, prob, f"Split 80/20 (n_test={len(yte)})"
    k = max(2, min(5, nneg, npos))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    try:
        prob = cross_val_predict(
            LogisticRegression(max_iter=2000), Xv_s, y,
            cv=skf, method="predict_proba")[:, 1]
    except Exception as e:
        print(f"  [WARN] cross_val_predict fallo ({e}); usando resubstitucion")
        clf  = LogisticRegression(max_iter=2000).fit(Xv_s, y)
        prob = clf.predict_proba(Xv_s)[:, 1]
        return y, prob, "Resubstitucion (n insuficiente)"
    return y, prob, f"Validacion cruzada estratificada {k}-fold"


def _prep_logit_df(df):
    """Construye (X_df, y) para regresion logistica."""
    cols = PREDICTORES_LOGIT + ["sedacion_6"]
    sub  = df[cols].dropna().copy()
    y    = sub["sedacion_6"].astype(int).values
    X    = sub[PREDICTORES_LOGIT].astype(float)
    return X, y, sub


def fig04_logit_roc(df):
    """Fig04: Curva ROC con AUC (grafico requerido 5)."""
    print("Generando Fig04 - Logit ROC...")
    X, y, _ = _prep_logit_df(df)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if len(y) < 10 or n_pos < 2 or n_neg < 2:
        _fig_vacia("P5_Fig04_LogReg_ROC.png",
                    f"Fig04: muestra insuficiente (n={len(y)}, +{n_pos}/-{n_neg})")
        return {}

    y_eval, prob, metodo = _eval_logistic_pred(X, y)
    if len(np.unique(y_eval)) < 2:
        _fig_vacia("P5_Fig04_LogReg_ROC.png",
                    "Fig04: una sola clase en evaluacion -> ROC no definida")
        return {}

    fpr, tpr, _ = roc_curve(y_eval, prob)
    auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.suptitle(f"FIGURA 4 - Regresion Logistica: Curva ROC — {COHORTE}",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)
    ax.plot(fpr, tpr, color=C1, lw=2.8, label=f"AUC = {auc_val:.3f}")
    ax.fill_between(fpr, 0, tpr, color=C1, alpha=0.15)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.2, label="Azar (AUC=0.5)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Tasa de falsos positivos (1 - Especificidad)", fontsize=10 * ESCALA_FUENTE)
    ax.set_ylabel("Tasa de verdaderos positivos (Sensibilidad)",   fontsize=10 * ESCALA_FUENTE)
    ax.set_title(f"Y = alcanza Ramsay 6  |  predictores: "
                  + ", ".join(LBL_VAR.get(v, v) for v in PREDICTORES_LOGIT)
                  + f"\nMetodo: {metodo}  |  n={len(y)} (+{n_pos}/-{n_neg})",
                  fontsize=9 * ESCALA_FUENTE, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10 * ESCALA_FUENTE, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out("P5_Fig04_LogReg_ROC.png"), facecolor=BG)
    plt.close(fig)
    print(f"[OK] Fig04 guardada | AUC={auc_val:.3f}")

    _add_resumen("Reg. logistica", "AUC-ROC (clasificacion Ramsay=6)",
                  f"AUC = {auc_val:.3f}", np.nan,
                  "Capacidad discriminativa "
                  + ("excelente" if auc_val >= 0.85 else
                     "buena" if auc_val >= 0.75 else
                     "moderada" if auc_val >= 0.6 else "limitada"))

    return {"[LogReg] AUC-ROC":
              {"test": "ROC", "stat": round(float(auc_val), 4),
               "p": np.nan, "sig": "—"}}


def fig05_logit_confusion(df):
    """Fig05: Matriz de confusion + metricas (grafico requerido 6)."""
    print("Generando Fig05 - Logit Matriz Confusion...")
    X, y, _ = _prep_logit_df(df)
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if len(y) < 10 or n_pos < 2 or n_neg < 2:
        _fig_vacia("P5_Fig05_LogReg_Matriz_Confusion.png",
                    f"Fig05: muestra insuficiente (n={len(y)}, +{n_pos}/-{n_neg})")
        return {}

    y_eval, prob, metodo = _eval_logistic_pred(X, y)
    y_pred = (prob >= 0.5).astype(int)
    cm     = confusion_matrix(y_eval, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    acc   = (tp + tn) / total if total else np.nan
    sens  = tp / (tp + fn) if (tp + fn) else np.nan
    spec  = tn / (tn + fp) if (tn + fp) else np.nan

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.suptitle(f"FIGURA 5 - Regresion Logistica: Matriz de Confusion — {COHORTE}",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                 xticklabels=["Pred. NO Ramsay=6", "Pred. SI Ramsay=6"],
                 yticklabels=["Real NO Ramsay=6",  "Real SI Ramsay=6"],
                 cbar=False, ax=ax, annot_kws={"fontsize": 18 * ESCALA_FUENTE, "fontweight": "bold"})
    ax.set_xlabel("Prediccion del modelo", fontsize=10 * ESCALA_FUENTE)
    ax.set_ylabel("Etiqueta real", fontsize=10 * ESCALA_FUENTE)
    titulo = (f"Umbral 0.5  |  Accuracy={acc:.2f}, Sensibilidad={sens:.2f}, "
              f"Especificidad={spec:.2f}\nMetodo: {metodo}")
    ax.set_title(titulo, fontsize=10 * ESCALA_FUENTE, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out("P5_Fig05_LogReg_Matriz_Confusion.png"), facecolor=BG)
    plt.close(fig)
    print(f"[OK] Fig05 guardada | Acc={acc:.2f} Sens={sens:.2f} Spec={spec:.2f}")

    _add_resumen("Reg. logistica", "Accuracy / Sensibilidad / Especificidad",
                  f"Acc={acc:.2f}, Sens={sens:.2f}, Spec={spec:.2f}",
                  np.nan,
                  "Rendimiento clasificador en evaluacion segun metodo aplicado")

    return {"[LogReg] Accuracy":     {"test": "—", "stat": round(float(acc), 4),
                                       "p": np.nan, "sig": "—"},
            "[LogReg] Sensibilidad": {"test": "—", "stat": round(float(sens), 4),
                                       "p": np.nan, "sig": "—"},
            "[LogReg] Especificidad":{"test": "—", "stat": round(float(spec), 4),
                                       "p": np.nan, "sig": "—"}}


def fig06_logit_forest(df):
    """Fig06: Forest plot de Odds Ratios con IC95% (grafico requerido 7)."""
    print("Generando Fig06 - Logit Forest Plot...")
    X, y, _ = _prep_logit_df(df)
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if len(y) < 10 or n_pos < 2 or n_neg < 2:
        _fig_vacia("P5_Fig06_LogReg_ForestPlot.png",
                    f"Fig06: muestra insuficiente (n={len(y)}, +{n_pos}/-{n_neg})")
        return {}

    res, metodo, _ = _fit_logit_inference(X, y)

    # Exportar CSV
    rows_csv = []
    for v, r in res.items():
        rows_csv.append({"Variable": LBL_VAR.get(v, v),
                          "Coef (log-OR)": r["coef"], "OR": r["OR"],
                          "IC95% inf": r["ci_lo"], "IC95% sup": r["ci_hi"],
                          "p-valor": r["p"]})
    pd.DataFrame(rows_csv).to_csv(out("P5_regresion_logistica.csv"),
                                    index=False, encoding="utf-8-sig",
                                    float_format="%.4f")

    fig, ax = plt.subplots(figsize=(10.5, max(7, 0.85 * len(res) + 2.3)))
    fig.suptitle(f"FIGURA 6 - Regresion Logistica: Odds Ratios + IC95% — {COHORTE}",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)

    ys = np.arange(len(res))[::-1]
    for i, (v, r) in enumerate(res.items()):
        y_pos = ys[i]
        or_, lo, hi, p = r["OR"], r["ci_lo"], r["ci_hi"], r["p"]
        # capear visualmente para evitar barras infinitas
        lo_c = max(lo, 1e-3) if not pd.isna(lo) else 1e-3
        hi_c = min(hi, 1e3)  if not pd.isna(hi) else 1e3
        or_c = max(min(or_, 1e3), 1e-3)
        is_sig = (not pd.isna(p)) and p < 0.05
        col = "#C0392B" if is_sig else "#34495E"
        ax.plot([lo_c, hi_c], [y_pos, y_pos], "-", color=col, lw=2.2)
        ax.plot([lo_c, lo_c], [y_pos - 0.15, y_pos + 0.15], "-", color=col, lw=2)
        ax.plot([hi_c, hi_c], [y_pos - 0.15, y_pos + 0.15], "-", color=col, lw=2)
        ax.plot(or_c, y_pos, "s", color=col, markersize=11,
                 markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        p_str = fmt_p(p) if not pd.isna(p) else "p=n/a"
        ax.text(hi_c * 1.18, y_pos,
                f"OR={or_:.2f}  [{lo:.2f}, {hi:.2f}]  {p_str}"
                if not pd.isna(lo) and not pd.isna(hi) else
                f"OR={or_:.2f}  [n/a]  {p_str}",
                va="center", fontsize=8.5 * ESCALA_FUENTE, color=col)

    ax.axvline(1.0, color="gray", ls="--", lw=1.3, alpha=0.85)
    ax.set_yticks(ys)
    ax.set_yticklabels([LBL_VAR.get(v, v) for v in res.keys()], fontsize=9 * ESCALA_FUENTE)
    ax.set_xscale("log")
    ax.set_xlabel("Odds Ratio (escala logaritmica)  ·  OR=1 -> sin efecto",
                   fontsize=10 * ESCALA_FUENTE)
    ax.set_title(f"Predictores de alcanzar Ramsay=6  |  {metodo}\n"
                  f"n={len(y)} (+{n_pos}/-{n_neg})  ·  Rojo = p<0.05",
                  fontsize=10 * ESCALA_FUENTE, fontweight="bold")
    # Margenes para etiquetas de la derecha
    cur_xlim = ax.get_xlim()
    ax.set_xlim(cur_xlim[0], cur_xlim[1] * 10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out("P5_Fig06_LogReg_ForestPlot.png"), facecolor=BG)
    plt.close(fig)
    print("[OK] Fig06 guardada | CSV: P5_regresion_logistica.csv")

    sig_vars = [LBL_VAR.get(v, v) for v, r in res.items()
                 if not pd.isna(r["p"]) and r["p"] < 0.05]
    _add_resumen("Reg. logistica",
                  "Predictores significativos de Ramsay=6 (p<0.05)",
                  ", ".join(sig_vars) if sig_vars else "Ninguno",
                  np.nan,
                  "Variables con OR significativamente distinto de 1"
                  if sig_vars else "Ningun predictor significativo (n limitado)")

    out_dict = {}
    for v, r in res.items():
        out_dict[f"[LogReg] OR {LBL_VAR.get(v, v)}"] = {
            "test": "Logit",
            "stat": round(float(r["OR"]), 4),
            "p":    round(float(r["p"]), 4) if not pd.isna(r["p"]) else np.nan,
            "sig":  sig_label(r["p"]),
        }
    return out_dict


# =============================================================================
# BLOQUE 2B - REGRESION LINEAL MULTIPLE  (Y = dosis_primera_ram6)
# =============================================================================

def _prep_lineal_df(df):
    cols = PREDICTORES_LINEAL + ["dosis_primera_ram6"]
    sub  = df[cols].dropna().copy()
    y    = sub["dosis_primera_ram6"].astype(float).values
    X    = sub[PREDICTORES_LINEAL].astype(float)
    return X, y, sub


def _fit_ols_full(X, y):
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc).fit()


def fig07_lineal_real_vs_pred(df):
    """Fig07: Real vs predicho (grafico requerido 8)."""
    print("Generando Fig07 - Lineal Real vs Predicho...")
    X, y, _ = _prep_lineal_df(df)
    if len(y) < 8:
        _fig_vacia("P5_Fig07_LinReg_Real_vs_Pred.png",
                    f"Fig07: muestra insuficiente (n={len(y)})")
        return {}

    # 80/20 split si n suficiente
    if len(y) >= 15:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42)
        m_tr = _fit_ols_full(Xtr, ytr)
        y_pred_te = m_tr.predict(sm.add_constant(Xte, has_constant="add"))
        y_pred_tr = m_tr.predict(sm.add_constant(Xtr, has_constant="add"))
        rmse_te = float(np.sqrt(np.mean((yte - y_pred_te) ** 2)))
        rmse_tr = float(np.sqrt(np.mean((ytr - y_pred_tr) ** 2)))
        split_lbl = f"Split 80/20  ·  RMSE_train={rmse_tr:.2f}  ·  RMSE_test={rmse_te:.2f}"
    else:
        m_tr = _fit_ols_full(X, y)
        y_pred_tr = m_tr.predict(sm.add_constant(X, has_constant="add"))
        ytr, Xtr = y, X
        yte, y_pred_te = np.array([]), np.array([])
        rmse_te = np.nan
        rmse_tr = float(np.sqrt(np.mean((y - y_pred_tr) ** 2)))
        split_lbl = f"In-sample  ·  RMSE={rmse_tr:.2f}  (n<15: sin split)"

    # Tambien fit full para R²
    m_full = _fit_ols_full(X, y)
    r2     = float(m_full.rsquared)
    r2adj  = float(m_full.rsquared_adj)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.suptitle(f"FIGURA 7 - Regresion Lineal: Real vs Predicho — {COHORTE}",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)
    ax.scatter(ytr, y_pred_tr, color=C2, alpha=0.75, s=80,
                edgecolors="white", lw=0.6, label=f"Entrenamiento (n={len(ytr)})")
    if len(yte) > 0:
        ax.scatter(yte, y_pred_te, color=C1, alpha=0.85, s=95,
                    edgecolors="white", lw=0.7, marker="D",
                    label=f"Test (n={len(yte)})")
    all_y = np.concatenate([ytr, yte]) if len(yte) > 0 else ytr
    all_p = np.concatenate([y_pred_tr, y_pred_te]) if len(yte) > 0 else y_pred_tr
    lo, hi = float(min(all_y.min(), all_p.min())), float(max(all_y.max(), all_p.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1.4, label="Identidad y=x")
    ax.set_xlabel("Dosis minima eficaz real (mg)", fontsize=10 * ESCALA_FUENTE)
    ax.set_ylabel("Dosis minima eficaz predicha (mg)", fontsize=10 * ESCALA_FUENTE)
    ax.set_title(f"R²={r2:.3f}  ·  R² ajustado={r2adj:.3f}\n{split_lbl}",
                  fontsize=10 * ESCALA_FUENTE, fontweight="bold")
    ax.legend(loc="best", fontsize=9 * ESCALA_FUENTE)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out("P5_Fig07_LinReg_Real_vs_Pred.png"), facecolor=BG)
    plt.close(fig)
    print(f"[OK] Fig07 guardada | R2={r2:.3f}, R2adj={r2adj:.3f}, RMSE_test={rmse_te if not pd.isna(rmse_te) else 'n/a'}")

    _add_resumen("Reg. lineal",
                  "R² ajustado (dosis minima eficaz)",
                  f"R²={r2:.3f} ; R²adj={r2adj:.3f}",
                  np.nan,
                  "Modelo " + ("informativo (R²>=0.3)" if r2 >= 0.3
                                else "con baja capacidad explicativa"))
    return {"[LinReg] R^2":       {"test": "OLS",
                                    "stat": round(r2, 4),
                                    "p": np.nan, "sig": "—"},
            "[LinReg] R^2 ajust.":{"test": "OLS",
                                    "stat": round(r2adj, 4),
                                    "p": np.nan, "sig": "—"},
            "[LinReg] RMSE test": {"test": "OLS",
                                    "stat": (round(rmse_te, 3)
                                              if not pd.isna(rmse_te) else np.nan),
                                    "p": np.nan, "sig": "—"}}


def fig08_lineal_residuos(df):
    """Fig08: Residuos vs ajustados + Q-Q plot (graficos requeridos 9 y 10)."""
    print("Generando Fig08 - Lineal Residuos + QQ...")
    X, y, _ = _prep_lineal_df(df)
    if len(y) < 6:
        _fig_vacia("P5_Fig08_LinReg_Residuos.png",
                    f"Fig08: muestra insuficiente (n={len(y)})")
        return {}

    m = _fit_ols_full(X, y)
    resid  = np.asarray(m.resid)
    fitted = np.asarray(m.fittedvalues)

    # Tests de supuestos
    try:
        sh_stat, sh_p = shapiro(resid)
    except Exception:
        sh_stat, sh_p = np.nan, np.nan
    try:
        bp = het_breuschpagan(resid, m.model.exog)
        bp_stat, bp_p = float(bp[0]), float(bp[1])
    except Exception:
        bp_stat, bp_p = np.nan, np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5))
    fig.suptitle(f"FIGURA 8 - Regresion Lineal: Diagnostico de Supuestos — {COHORTE}",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)

    # Panel A: residuos vs ajustados
    ax = axes[0]
    ax.scatter(fitted, resid, color=C2, s=70, alpha=0.75,
                edgecolors="white", lw=0.6)
    ax.axhline(0, color="#E74C3C", ls="--", lw=1.4)
    # Suavizado (LOWESS si hay statsmodels)
    if _HAS_LOWESS and len(fitted) >= 10:
        try:
            sm_l = sm_lowess(resid, fitted, frac=0.6, return_sorted=True)
            ax.plot(sm_l[:, 0], sm_l[:, 1], color=CGLOBAL, lw=2,
                     alpha=0.85, label="LOWESS")
            ax.legend(fontsize=8 * ESCALA_FUENTE)
        except Exception:
            pass
    ax.set_xlabel("Valor ajustado (predicho)", fontsize=9 * ESCALA_FUENTE)
    ax.set_ylabel("Residuo", fontsize=9 * ESCALA_FUENTE)
    ax.set_title(f"A · Residuos vs Ajustados\n"
                  f"Breusch-Pagan p={fmt_p(bp_p)} (homocedasticidad)",
                  fontsize=10 * ESCALA_FUENTE, fontweight="bold")

    # Panel B: Q-Q plot
    ax = axes[1]
    (osm, osr), (slope, intercept, r_qq) = probplot(resid, dist="norm")
    ax.scatter(osm, osr, color=C1, s=65, alpha=0.8,
                edgecolors="white", lw=0.6)
    ax.plot(osm, slope * osm + intercept, "--", color=CGLOBAL, lw=1.4,
             label="Linea normal teorica")
    ax.set_xlabel("Cuantiles teoricos (normal)", fontsize=9 * ESCALA_FUENTE)
    ax.set_ylabel("Cuantiles observados (residuos)", fontsize=9 * ESCALA_FUENTE)
    ax.set_title(f"B · Q-Q plot de residuos\n"
                  f"Shapiro-Wilk p={fmt_p(sh_p)} (normalidad)",
                  fontsize=10 * ESCALA_FUENTE, fontweight="bold")
    ax.legend(fontsize=8 * ESCALA_FUENTE)

    # Pie con interpretacion
    interp = []
    interp.append("Normalidad: " +
                   ("OK (Shapiro p>0.05)" if not pd.isna(sh_p) and sh_p > 0.05
                    else ("VIOLADA" if not pd.isna(sh_p) else "n/c")))
    interp.append("Homocedasticidad: " +
                   ("OK (BP p>0.05)" if not pd.isna(bp_p) and bp_p > 0.05
                    else ("VIOLADA" if not pd.isna(bp_p) else "n/c")))
    fig.text(0.5, 0.005, "  |  ".join(interp), ha="center",
              fontsize=8.5 * ESCALA_FUENTE, color="gray", fontweight="bold")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out("P5_Fig08_LinReg_Residuos.png"), facecolor=BG)
    plt.close(fig)
    print(f"[OK] Fig08 guardada | Shapiro p={sh_p}, BP p={bp_p}")

    _add_resumen("Reg. lineal", "Supuesto normalidad residuos (Shapiro)",
                  fmt_p(sh_p), sh_p,
                  "OK" if not pd.isna(sh_p) and sh_p > 0.05 else "Violada (revisar transformacion)")
    _add_resumen("Reg. lineal", "Supuesto homocedasticidad (Breusch-Pagan)",
                  fmt_p(bp_p), bp_p,
                  "OK" if not pd.isna(bp_p) and bp_p > 0.05 else "Violada (varianza no constante)")

    return {"[LinReg] Shapiro residuos":   {"test": "Shapiro-Wilk",
                                              "stat": round(float(sh_stat), 4) if not pd.isna(sh_stat) else np.nan,
                                              "p": round(float(sh_p), 4) if not pd.isna(sh_p) else np.nan,
                                              "sig": sig_label(sh_p)},
            "[LinReg] Breusch-Pagan":      {"test": "Breusch-Pagan",
                                              "stat": round(float(bp_stat), 4) if not pd.isna(bp_stat) else np.nan,
                                              "p": round(float(bp_p), 4) if not pd.isna(bp_p) else np.nan,
                                              "sig": sig_label(bp_p)}}


def fig09_lineal_importancia(df):
    """Fig09: Importancia de variables (coeficientes estandarizados) (graf. 11)."""
    print("Generando Fig09 - Lineal Importancia Variables...")
    X, y, _ = _prep_lineal_df(df)
    if len(y) < 6:
        _fig_vacia("P5_Fig09_LinReg_Importancia.png",
                    f"Fig09: muestra insuficiente (n={len(y)})")
        return {}

    # Coeficientes estandarizados: refit con X z-scored e y z-scored
    Xs = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    ys = (y - y.mean()) / (y.std(ddof=0) if y.std(ddof=0) > 0 else 1)
    m_std = sm.OLS(ys, sm.add_constant(Xs, has_constant="add")).fit()

    # Coef raw para tabla y CSV
    m_full = _fit_ols_full(X, y)

    rows_csv = []
    for v in X.columns:
        rows_csv.append({
            "Variable":            LBL_VAR.get(v, v),
            "Coef (no estand.)":   float(m_full.params[v]),
            "Coef estandarizado":  float(m_std.params[v]),
            "IC95% inf":           float(m_full.conf_int().loc[v, 0]),
            "IC95% sup":           float(m_full.conf_int().loc[v, 1]),
            "p-valor":             float(m_full.pvalues[v]),
        })
    pd.DataFrame(rows_csv).to_csv(out("P5_regresion_lineal.csv"),
                                    index=False, encoding="utf-8-sig",
                                    float_format="%.4f")

    # Ordenar por |coef estandarizado|
    items = sorted([(v, float(m_std.params[v]), float(m_full.pvalues[v]))
                    for v in X.columns],
                   key=lambda t: abs(t[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(9.5, max(6.5, 0.85 * len(items) + 2.3)))
    fig.suptitle(f"FIGURA 9 - Regresion Lineal: Importancia de Variables — {COHORTE}",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)

    labels = [LBL_VAR.get(v, v) for v, _, _ in items]
    vals   = [b for _, b, _ in items]
    ps     = [p for _, _, p in items]
    ys_pos = np.arange(len(items))[::-1]
    cols   = ["#C0392B" if (not pd.isna(p) and p < 0.05) else "#1A5276"
              for p in ps]

    bars = ax.barh(ys_pos, vals, color=cols, alpha=0.85, edgecolor="white")
    for b, p in zip(bars, ps):
        w = b.get_width()
        ax.text(w + 0.02 * np.sign(w if w != 0 else 1),
                b.get_y() + b.get_height() / 2,
                fmt_p(p), va="center",
                ha="left" if w >= 0 else "right",
                fontsize=8 * ESCALA_FUENTE, color=CGLOBAL)
    ax.axvline(0, color="gray", lw=1)
    ax.set_yticks(ys_pos); ax.set_yticklabels(labels, fontsize=9 * ESCALA_FUENTE)
    ax.set_xlabel("Coeficiente estandarizado (beta)", fontsize=10 * ESCALA_FUENTE)
    ax.set_title("Ordenado por |beta|  ·  Rojo = p<0.05",
                  fontsize=10 * ESCALA_FUENTE, fontweight="bold")
    # Margen extra a ambos lados: las etiquetas de p-valor se extienden mas
    # alla de la punta de la barra y, sin este hueco, invaden las categorias
    # del eje Y (barras negativas) o se salen del eje (barras positivas).
    x_lo, x_hi = ax.get_xlim()
    pad = max(abs(x_lo), abs(x_hi)) * 0.55
    ax.set_xlim(x_lo - pad, x_hi + pad)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out("P5_Fig09_LinReg_Importancia.png"), facecolor=BG)
    plt.close(fig)
    print("[OK] Fig09 guardada | CSV: P5_regresion_lineal.csv")

    top_pred = items[0][0] if items else None
    _add_resumen("Reg. lineal", "Predictor mas influyente (|beta|)",
                  LBL_VAR.get(top_pred, top_pred) if top_pred else "n/a",
                  items[0][2] if items else np.nan,
                  "Variable con mayor peso estandarizado sobre la dosis minima eficaz")

    out_dict = {}
    for v in X.columns:
        out_dict[f"[LinReg] beta {LBL_VAR.get(v, v)}"] = {
            "test": "OLS",
            "stat": round(float(m_std.params[v]), 4),
            "p":    round(float(m_full.pvalues[v]), 4),
            "sig":  sig_label(float(m_full.pvalues[v])),
        }
    return out_dict


# =============================================================================
# BLOQUE 2C - PRUEBAS DE HIPOTESIS
# =============================================================================

def fig10_hipotesis_boxplots(df):
    """Fig10: Boxplots comparativos Onco vs No-Onco con anotaciones p-valor
    (grafico requerido 12)."""
    print("Generando Fig10 - Boxplots de Hipotesis...")
    vars_use = [v for v in VARS_NUM_HIP if v in df.columns]
    if not vars_use:
        _fig_vacia("P5_Fig10_Hipotesis_Boxplots.png",
                    "Fig10: variables no disponibles")
        return {}

    n_cols = 4
    n_rows = int(np.ceil(len(vars_use) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6.5 * n_rows))
    fig.suptitle(f"FIGURA 10 - Pruebas de Hipotesis: Oncologico vs No Oncologico — {COHORTE}",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)
    axes = np.array(axes).reshape(n_rows, n_cols)
    stats_out = {}

    for i, v in enumerate(vars_use):
        ax = axes[i // n_cols, i % n_cols]
        g1 = df[df["Tipo de paciente"] == 1][v].dropna()
        g2 = df[df["Tipo de paciente"] == 2][v].dropna()

        if len(g1) < 2 or len(g2) < 2:
            ax.text(0.5, 0.5, f"n insuficiente\n(g1={len(g1)}, g2={len(g2)})",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9 * ESCALA_FUENTE, color="gray")
            ax.set_title(LBL_VAR.get(v, v), fontsize=10 * ESCALA_FUENTE, fontweight="bold")
            ax.set_xticks([])
            continue

        test_name, stat_v, p_v = _auto_test_2groups(g1, g2)
        bp = ax.boxplot([g1.values, g2.values], positions=[1, 2],
                         patch_artist=True, widths=0.45,
                         medianprops=dict(color="white", lw=2),
                         whiskerprops=dict(color=CGLOBAL),
                         capprops=dict(color=CGLOBAL),
                         flierprops=dict(marker="o", markersize=3, alpha=0.4))
        for patch, c in zip(bp["boxes"], [C1, C2]):
            patch.set_facecolor(c); patch.set_alpha(0.78)
        for j, (g, c) in enumerate([(g1, C1), (g2, C2)], 1):
            jit = np.random.uniform(-0.15, 0.15, size=len(g))
            ax.scatter(np.full(len(g), j) + jit, g.values, color=c,
                        alpha=0.55, s=30, edgecolors="white", lw=0.4, zorder=3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"Onco\n(n={len(g1)})",
                              f"No-Onco\n(n={len(g2)})"], fontsize=9 * ESCALA_FUENTE)
        ax.set_ylabel(LBL_VAR.get(v, v), fontsize=9.5 * ESCALA_FUENTE)
        # corchete p
        y_max = max(float(g1.max()), float(g2.max()))
        ax.set_ylim(top=y_max * 1.18 + (0.5 if y_max < 5 else 0))
        _annot_pval(ax, 1, 2, y_max * 1.05, p_v)
        ax.set_title(f"{LBL_VAR.get(v, v)}\n{test_name}",
                      fontsize=10 * ESCALA_FUENTE, fontweight="bold")
        stats_out[f"[Hipotesis] {LBL_VAR.get(v, v)} (Onco vs NoOnco)"] = {
            "test": test_name,
            "stat": round(float(stat_v), 3) if not pd.isna(stat_v) else np.nan,
            "p":    round(float(p_v), 4) if not pd.isna(p_v) else np.nan,
            "sig":  sig_label(p_v),
        }

    # ocultar paneles sobrantes
    for k in range(len(vars_use), n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.92], h_pad=3.0, w_pad=2.2)
    fig.savefig(out("P5_Fig10_Hipotesis_Boxplots.png"), facecolor=BG)
    plt.close(fig)
    print("[OK] Fig10 guardada")

    sig_vars = [k for k, vv in stats_out.items()
                 if vv["p"] is not np.nan and not pd.isna(vv["p"]) and vv["p"] < 0.05]
    _add_resumen("Hipotesis", "Variables con diferencia Onco vs No-Onco (p<0.05)",
                  ", ".join([k.split("] ")[1].split(" (")[0] for k in sig_vars])
                  if sig_vars else "Ninguna",
                  np.nan,
                  "Tests parametricos o no parametricos segun Shapiro")
    return stats_out


def fig11_hipotesis_tabla(df):
    """Fig11: Tabla resumen de tests aplicados (grafico requerido 13)."""
    print("Generando Fig11 - Tabla Tests Hipotesis...")
    vars_use = [v for v in VARS_NUM_HIP if v in df.columns]

    filas = []
    for v in vars_use:
        # Onco vs No-Onco
        g1 = df[df["Tipo de paciente"] == 1][v].dropna()
        g2 = df[df["Tipo de paciente"] == 2][v].dropna()
        try:
            sh1 = shapiro(g1)[1] if len(g1) >= 3 else np.nan
            sh2 = shapiro(g2)[1] if len(g2) >= 3 else np.nan
        except Exception:
            sh1, sh2 = np.nan, np.nan
        norm_lbl = f"Onco={fmt_p(sh1)}\nNo-Onco={fmt_p(sh2)}"
        tname, st, pv = _auto_test_2groups(g1, g2)
        filas.append([LBL_VAR.get(v, v), "Onco vs No-Onco",
                       f"n={len(g1)}/{len(g2)}", norm_lbl, tname,
                       _fmt_stat(tname, st), fmt_p(pv), sig_label(pv)])

        # Con vs Sin evento adverso (solo si hay grupos)
        g_si = df[df["evento_adverso"] == 1][v].dropna()
        g_no = df[df["evento_adverso"] == 0][v].dropna()
        if len(g_si) >= 2 and len(g_no) >= 2:
            try:
                shs = shapiro(g_si)[1] if len(g_si) >= 3 else np.nan
                shn = shapiro(g_no)[1] if len(g_no) >= 3 else np.nan
            except Exception:
                shs, shn = np.nan, np.nan
            norm_lbl = f"Si EA={fmt_p(shs)}\nSin EA={fmt_p(shn)}"
            tn2, st2, pv2 = _auto_test_2groups(g_si, g_no)
            filas.append([LBL_VAR.get(v, v), "Con vs Sin evento adverso",
                           f"n={len(g_si)}/{len(g_no)}", norm_lbl, tn2,
                           _fmt_stat(tn2, st2), fmt_p(pv2), sig_label(pv2)])

    # Kruskal por cuartiles de dosis acumulada para 3 variables clave
    if df["dosis_acum_mid"].notna().sum() >= 8:
        sub = df.dropna(subset=["dosis_acum_mid"]).copy()
        try:
            sub["q_dosis"] = pd.qcut(sub["dosis_acum_mid"], q=3,
                                       labels=["T1", "T2", "T3"],
                                       duplicates="drop")
            for v in ["sedacion_6", "n_rescates_e2", "tiempo_fallec_dias"]:
                if v not in sub.columns: continue
                grps = [sub[sub["q_dosis"] == q][v].dropna().values
                        for q in sub["q_dosis"].cat.categories]
                grps = [g for g in grps if len(g) >= 2]
                if len(grps) >= 2:
                    tn3, st3, pv3 = _auto_test_kgroups(grps)
                    filas.append([LBL_VAR.get(v, v),
                                    "Terciles dosis acumulada",
                                    f"n_grupos={len(grps)}", "—",
                                    tn3, _fmt_stat(tn3, st3),
                                    fmt_p(pv3), sig_label(pv3)])
        except Exception as e:
            print(f"  [WARN] Kruskal cuartiles fallo: {e}")

    cols_t = ["Variable", "Comparacion", "n", "Normalidad (Shapiro p)",
               "Test", "Estadistico", "p-valor", "Sig."]
    pd.DataFrame(filas, columns=cols_t).to_csv(
        out("P5_tabla_tests_hipotesis.csv"),
        index=False, encoding="utf-8-sig")

    n_rows = len(filas) + 1
    fig_h  = max(7, n_rows * 1.1 + 3.0)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    fig.suptitle(f"FIGURA 11 - Resumen de Tests de Hipotesis — {COHORTE}",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)

    tbl = ax.table(cellText=filas, colLabels=cols_t, cellLoc="center",
                    bbox=[0.0, 0.05, 1.0, 0.88])
    tbl.auto_set_font_size(False)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(CGLOBAL)
            cell.set_text_props(color="white", fontweight="bold", fontsize=12 * ESCALA_FUENTE)
        else:
            cell.set_text_props(fontsize=11 * ESCALA_FUENTE)
            if c == 7:  # columna sig
                v = filas[r - 1][7]
                cell.set_facecolor("#FADBD8" if v in ("*", "**") else "#E8F8F5")
        cell.set_edgecolor("#CCCCCC")

    # auto_set_column_width se llama despues de fijar la fuente de cada
    # celda (y tras forzar un draw), para que los anchos se calculen con
    # el tamano de fuente real y no con el valor por defecto de la tabla.
    fig.canvas.draw()
    tbl.auto_set_column_width(col=list(range(len(cols_t))))

    fig.tight_layout()
    fig.savefig(out("P5_Fig11_Hipotesis_Tabla_Tests.png"), facecolor=BG)
    plt.close(fig)
    print("[OK] Fig11 guardada | CSV: P5_tabla_tests_hipotesis.csv")
    return {}


# =============================================================================
# BLOQUE 3A - SUBGRUPO POR DIAGNOSTICO
# =============================================================================

def fig12_subgrupo_diagnostico(df):
    """Fig12: 3 paneles (graficos requeridos 14, 15, 16)."""
    print("Generando Fig12 - Subgrupo Diagnostico...")
    sub = df.dropna(subset=["Tipo de paciente"])
    stats_out = {}

    fig = plt.figure(figsize=(13, 8.5))
    fig.suptitle(f"FIGURA 12 - Subgrupo: Diagnostico Oncologico vs No Oncologico — {COHORTE}",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.65)

    # ── A: % sedacion 6 alcanzada por grupo (barplot con IC95) ───────────
    ax = fig.add_subplot(gs[0, 0])
    pcts, ns, Ns = [], [], []
    for t in (1, 2):
        s = sub[sub["Tipo de paciente"] == t]["sedacion_6"].dropna()
        n_si = int((s == 1).sum()); N = len(s)
        pct, lo, hi = prop_ic95(n_si, N)
        pcts.append((pct, lo, hi)); ns.append(n_si); Ns.append(N)
    xs   = [0, 1]
    bars = ax.bar(xs, [p[0] for p in pcts], color=[C1, C2], alpha=0.82,
                   edgecolor="white", width=0.55)
    e_lo = [p[0] - p[1] for p in pcts]; e_hi = [p[2] - p[0] for p in pcts]
    ax.errorbar(xs, [p[0] for p in pcts], yerr=[e_lo, e_hi], fmt="none",
                 color=CGLOBAL, capsize=6, lw=1.5)
    for i, (b, p, n_, N_) in enumerate(zip(bars, pcts, ns, Ns)):
        ax.text(b.get_x() + b.get_width() / 2, p[0] + 3,
                f"{p[0]:.1f}%\n(n={n_}/{N_})\nIC95%[{p[1]:.0f}-{p[2]:.0f}]",
                ha="center", fontsize=8 * ESCALA_FUENTE)
    # test chi2/Fisher
    if Ns[0] > 0 and Ns[1] > 0:
        ct = np.array([[ns[0], Ns[0] - ns[0]], [ns[1], Ns[1] - ns[1]]])
        try:
            tn, tv, pv = chi2_or_fisher(ct)
        except Exception:
            tn, tv, pv = "—", np.nan, np.nan
    else:
        tn, tv, pv = "—", np.nan, np.nan
    ax.set_xticks(xs); ax.set_xticklabels([LABEL_CASO[1].replace(" - ", "\n"),
                                            LABEL_CASO[2].replace(" - ", "\n")],
                                            fontsize=9 * ESCALA_FUENTE)
    ax.set_ylabel("% pacientes con sedacion = 6 (Ramsay)", fontsize=9 * ESCALA_FUENTE)
    ax.set_ylim(0, 125)
    ax.set_title(f"A · % Sedacion=6 alcanzada\n"
                  f"{tn} ({_fmt_stat(tn, tv)}) p={fmt_p(pv)}",
                  fontsize=10 * ESCALA_FUENTE, fontweight="bold")
    stats_out["[Subgrupo] Sedacion=6 Onco vs NoOnco"] = {
        "test": tn, "stat": tv,
        "p": round(float(pv), 4) if not pd.isna(pv) else np.nan,
        "sig": sig_label(pv),
    }
    _add_resumen("Subgrupos · Diagnostico",
                  "Diferencia en tasa de sedacion=6 entre grupos",
                  f"Onco={pcts[0][0]:.0f}% vs No-Onco={pcts[1][0]:.0f}%",
                  pv,
                  ("Diferencia significativa" if not pd.isna(pv) and pv < 0.05
                   else "Sin diferencia significativa"))

    # ── B: Violinplot dosis minima eficaz por diagnostico ────────────────
    ax = fig.add_subplot(gs[0, 1])
    g1 = sub[sub["Tipo de paciente"] == 1]["dosis_primera_ram6"].dropna()
    g2 = sub[sub["Tipo de paciente"] == 2]["dosis_primera_ram6"].dropna()
    if len(g1) >= 2 and len(g2) >= 2:
        vp = ax.violinplot([g1.values, g2.values], positions=[1, 2],
                            showmedians=True, showextrema=True, widths=0.6)
        for body, c in zip(vp["bodies"], [C1, C2]):
            body.set_facecolor(c); body.set_alpha(0.55)
            body.set_edgecolor(CGLOBAL)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            if part in vp:
                vp[part].set_edgecolor(CGLOBAL)
        for i, (g, c) in enumerate([(g1, C1), (g2, C2)], 1):
            jit = np.random.uniform(-0.13, 0.13, size=len(g))
            ax.scatter(np.full(len(g), i) + jit, g.values, color=c,
                        alpha=0.7, s=45, edgecolors="white", lw=0.5, zorder=3)
            ax.text(i, float(g.max()) * 1.05, f"Med={g.median():.0f}", va="bottom",
                    ha="center", fontsize=8 * ESCALA_FUENTE, color=c, fontweight="bold")
        max_b = max(float(g1.max()), float(g2.max()))
        ax.set_ylim(top=max_b * 1.28)
        tn2, st2, pv2 = _auto_test_2groups(g1, g2)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"Onco (n={len(g1)})",
                              f"No-Onco (n={len(g2)})"], fontsize=9 * ESCALA_FUENTE)
        ax.set_ylabel("Dosis minima eficaz (mg)", fontsize=9 * ESCALA_FUENTE)
        ax.set_title(f"B · Dosis minima eficaz\n"
                      f"{tn2} p={fmt_p(pv2)}",
                      fontsize=10 * ESCALA_FUENTE, fontweight="bold")
        stats_out["[Subgrupo] Dosis min Onco vs NoOnco"] = {
            "test": tn2,
            "stat": round(float(st2), 3) if not pd.isna(st2) else np.nan,
            "p":    round(float(pv2), 4) if not pd.isna(pv2) else np.nan,
            "sig":  sig_label(pv2),
        }
    else:
        ax.text(0.5, 0.5, "n insuficiente", ha="center", va="center",
                transform=ax.transAxes, fontsize=10 * ESCALA_FUENTE, color="gray")
        ax.set_title("B · Dosis minima eficaz",
                      fontsize=10 * ESCALA_FUENTE, fontweight="bold")

    # ── C: Eventos adversos por diagnostico (barras apiladas) ────────────
    ax = fig.add_subplot(gs[0, 2])
    grupos = [1, 2]
    n_si_e = [int((sub[(sub["Tipo de paciente"] == t) &
                          (sub["evento_adverso"] == 1)]).shape[0])
              for t in grupos]
    n_no_e = [int((sub[(sub["Tipo de paciente"] == t) &
                          (sub["evento_adverso"] == 0)]).shape[0])
              for t in grupos]
    totals = [n_si_e[i] + n_no_e[i] for i in range(2)]
    if all(t > 0 for t in totals):
        pct_si = [n_si_e[i] / totals[i] * 100 for i in range(2)]
        pct_no = [100 - pct_si[i] for i in range(2)]
        xs2 = [0, 1]
        ax.bar(xs2, pct_no, color=PAL_OK[0], alpha=0.85, label="Sin evento adverso",
                edgecolor="white", width=0.55)
        ax.bar(xs2, pct_si, bottom=pct_no, color=PAL_ALERTA[0], alpha=0.9,
                label="Con evento adverso", edgecolor="white", width=0.55)
        for i in range(2):
            ax.text(i, pct_no[i] / 2, f"{pct_no[i]:.0f}%\n(n={n_no_e[i]})",
                    ha="center", va="center", color="white",
                    fontsize=8 * ESCALA_FUENTE, fontweight="bold")
            ax.text(i, pct_no[i] + pct_si[i] / 2,
                    f"{pct_si[i]:.0f}%\n(n={n_si_e[i]})",
                    ha="center", va="center", color="white",
                    fontsize=8 * ESCALA_FUENTE, fontweight="bold")
        ct = np.array([[n_si_e[0], n_no_e[0]], [n_si_e[1], n_no_e[1]]])
        try:
            tn3, tv3, pv3 = chi2_or_fisher(ct)
        except Exception:
            tn3, tv3, pv3 = "—", np.nan, np.nan
        ax.set_xticks(xs2); ax.set_xticklabels([LABEL_CASO[1].replace(" - ", "\n"),
                                                 LABEL_CASO[2].replace(" - ", "\n")],
                                                fontsize=9 * ESCALA_FUENTE)
        ax.set_ylabel("% pacientes", fontsize=9 * ESCALA_FUENTE)
        ax.set_ylim(0, 105)
        ax.set_title(f"C · Eventos adversos\n"
                      f"{tn3} ({_fmt_stat(tn3, tv3)}) p={fmt_p(pv3)}",
                      fontsize=10 * ESCALA_FUENTE, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8 * ESCALA_FUENTE, framealpha=0.95)
        stats_out["[Subgrupo] EA Onco vs NoOnco"] = {
            "test": tn3, "stat": tv3,
            "p": round(float(pv3), 4) if not pd.isna(pv3) else np.nan,
            "sig": sig_label(pv3),
        }
    else:
        ax.text(0.5, 0.5, "Datos insuficientes", ha="center", va="center",
                transform=ax.transAxes, fontsize=10 * ESCALA_FUENTE, color="gray")
        ax.set_title("C · Eventos adversos",
                      fontsize=10 * ESCALA_FUENTE, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.88], w_pad=2.8)
    fig.savefig(out("P5_Fig12_Subgrupo_Diagnostico.png"), facecolor=BG)
    plt.close(fig)
    print("[OK] Fig12 guardada")
    return stats_out


# =============================================================================
# BLOQUE 3B - SUBGRUPO POR RANGO DE DOSIS (CUARTILES)
# =============================================================================

def fig13_subgrupo_dosis(df):
    """Fig13: tasa sedacion 6 por cuartil + scatter dosis vs sedacion
    (graficos requeridos 17 y 18)."""
    print("Generando Fig13 - Subgrupo Dosis...")
    valid = df.dropna(subset=["dosis_acum_mid"]).copy()
    if len(valid) < 8:
        _fig_vacia("P5_Fig13_Subgrupo_Dosis.png",
                    "Fig13: muestra insuficiente para cuartiles de dosis")
        return {}

    # Construir cuartiles (qcut con fallback a bins fijos)
    usar_fijos = False
    if valid["dosis_acum_mid"].nunique() >= 3:
        try:
            valid["q"] = pd.qcut(valid["dosis_acum_mid"], q=3,
                                  labels=["T1", "T2", "T3"],
                                  duplicates="drop")
            if valid["q"].nunique() < 3:
                usar_fijos = True
        except Exception:
            usar_fijos = True
    else:
        usar_fijos = True
    if usar_fijos:
        valid["q"] = pd.cut(valid["dosis_acum_mid"],
                              bins=[-np.inf, 40, 60, np.inf],
                              labels=["<=40 mg", "50-60 mg", ">=70 mg"])

    q_labels = [q for q in valid["q"].cat.categories
                 if (valid["q"] == q).sum() > 0]
    rangos   = {q: valid[valid["q"] == q]["dosis_acum_mid"].agg(["min", "max"])
                 for q in q_labels}

    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5))
    fig.suptitle(f"FIGURA 13 - Subgrupo: Rango de Dosis Acumulada — {COHORTE}",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)
    stats_out = {}

    # ── A: % sedacion 6 por cuartil (lineplot + barras) ──────────────────
    ax = axes[0]
    pcts, los, his, ns_tot, ns_si = [], [], [], [], []
    for q in q_labels:
        s    = valid[valid["q"] == q]["sedacion_6"].dropna()
        n_si = int((s == 1).sum()); N = len(s)
        pct, lo, hi = prop_ic95(n_si, N)
        pcts.append(pct); los.append(lo); his.append(hi)
        ns_tot.append(N); ns_si.append(n_si)
    xs = np.arange(len(q_labels))
    bars = ax.bar(xs, pcts, color=PAL_GRAD[:len(q_labels)], alpha=0.8,
                   edgecolor="white", width=0.55)
    ax.plot(xs, pcts, "o-", color=CGLOBAL, lw=2.4, markersize=9, zorder=4)
    ax.errorbar(xs, pcts,
                 yerr=[[p - l for p, l in zip(pcts, los)],
                       [h - p for p, h in zip(pcts, his)]],
                 fmt="none", color=CGLOBAL, capsize=5, lw=1.4)
    for b, p, n_si, N in zip(bars, pcts, ns_si, ns_tot):
        ax.text(b.get_x() + b.get_width() / 2, p + 4,
                f"{p:.0f}%\n({n_si}/{N})", ha="center", fontsize=8 * ESCALA_FUENTE)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{q}\n({rangos[q]['min']:.0f}-{rangos[q]['max']:.0f} mg)"
                          for q in q_labels], fontsize=8 * ESCALA_FUENTE)
    ax.set_ylabel("% pacientes con sedacion = 6", fontsize=9 * ESCALA_FUENTE)
    ax.set_ylim(0, 130)

    # Kruskal-Wallis (o equivalente) sobre sedacion_6 entre cuartiles
    grps = [valid[valid["q"] == q]["sedacion_6"].dropna().values
            for q in q_labels]
    grps = [g for g in grps if len(g) >= 2]
    tnK, stK, pvK = ("—", np.nan, np.nan)
    if len(grps) >= 2:
        try:
            tnK, stK, pvK = _auto_test_kgroups(grps)
        except Exception:
            pass
    ax.set_title(f"A · Tasa de sedacion=6 por tercil de dosis acumulada\n"
                  f"{tnK} p={fmt_p(pvK)}", fontsize=10 * ESCALA_FUENTE, fontweight="bold")
    stats_out["[Subgrupo] Sedacion=6 por cuartil dosis"] = {
        "test": tnK,
        "stat": round(float(stK), 3) if not pd.isna(stK) else np.nan,
        "p":    round(float(pvK), 4) if not pd.isna(pvK) else np.nan,
        "sig":  sig_label(pvK),
    }
    _add_resumen("Subgrupos · Dosis",
                  "Relacion dosis-respuesta (terciles)",
                  " > ".join(f"{q}={p:.0f}%" for q, p in zip(q_labels, pcts)),
                  pvK,
                  "Relacion dosis-respuesta detectada"
                  if not pd.isna(pvK) and pvK < 0.05
                  else "Sin gradiente significativo entre terciles")

    # ── B: Scatter dosis acumulada vs Ramsay final + tendencia ───────────
    # Mejoras visuales: jitter vertical para revelar pacientes solapados en
    # el mismo nivel Ramsay (efecto techo en Ramsay=6) y marcadores
    # diferenciados para distinguir quienes alcanzaron Ramsay=6.
    ax = axes[1]
    vsc = valid.dropna(subset=["ramsay_ultima"]).copy()
    rng_j = np.random.default_rng(42)
    vsc["_ry"] = vsc["ramsay_ultima"] + rng_j.uniform(-0.13, 0.13, size=len(vsc))
    for t in (1, 2):
        sub_t = vsc[vsc["Tipo de paciente"] == t]
        if len(sub_t) == 0: continue
        m_si = sub_t["ramsay_ultima"] == 6
        if m_si.any():
            ax.scatter(sub_t.loc[m_si, "dosis_acum_mid"],
                        sub_t.loc[m_si, "_ry"],
                        color=COLOR_CASO[t],
                        label=f"{LABEL_CASO[t]} (R=6)",
                        s=75, alpha=0.78, edgecolors="white", lw=0.5,
                        marker="o", zorder=3)
        m_no = sub_t["ramsay_ultima"] != 6
        if m_no.any():
            ax.scatter(sub_t.loc[m_no, "dosis_acum_mid"],
                        sub_t.loc[m_no, "_ry"],
                        color=COLOR_CASO[t],
                        label=f"{LABEL_CASO[t]} (R<6)",
                        s=145, alpha=0.95, edgecolors=CGLOBAL, lw=1.3,
                        marker="X", zorder=5)
    if len(vsc) >= 5:
        xs = vsc["dosis_acum_mid"].values
        ys = vsc["ramsay_ultima"].values
        if _HAS_LOWESS and len(vsc) >= 10:
            try:
                lo_ = sm_lowess(ys, xs, frac=0.6, return_sorted=True)
                ax.plot(lo_[:, 0], lo_[:, 1], color=CGLOBAL, lw=2.4,
                         label="LOWESS")
            except Exception:
                pass
        else:
            slope, inter, *_ = stats.linregress(xs, ys)
            xf = np.linspace(xs.min(), xs.max(), 200)
            ax.plot(xf, slope * xf + inter, "-", color=CGLOBAL, lw=2,
                     label="Regresion lineal")
        rho, p_rho = stats.spearmanr(xs, ys)
        ax.text(0.02, 0.98, f"Spearman r={rho:.3f}\np={fmt_p(p_rho)}",
                transform=ax.transAxes, va="top", fontsize=9 * ESCALA_FUENTE,
                bbox=dict(boxstyle="round", facecolor="white",
                           edgecolor="#BDC3C7", alpha=0.9))
        stats_out["[Subgrupo] Spearman dosis-Ramsay"] = {
            "test": "Spearman",
            "stat": round(float(rho), 4) if not pd.isna(rho) else np.nan,
            "p":    round(float(p_rho), 4) if not pd.isna(p_rho) else np.nan,
            "sig":  sig_label(p_rho),
        }
    ax.axhline(6, color="#27AE60", ls="--", lw=1.3, alpha=0.6,
                label="Objetivo Ramsay=6")
    ax.set_xlabel("Dosis acumulada de Midazolam (mg)", fontsize=9 * ESCALA_FUENTE)
    ax.set_ylabel("Ramsay alcanzado (ultima visita)", fontsize=9 * ESCALA_FUENTE)
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_ylim(0.7, 6.5)
    n_no_r6 = int((vsc["ramsay_ultima"] != 6).sum())
    ax.set_title(f"B · Dosis acumulada vs Ramsay final\n"
                  f"(jitter vertical; X = no alcanzo R=6: n={n_no_r6})",
                  fontsize=10 * ESCALA_FUENTE, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7.5 * ESCALA_FUENTE, ncol=2, framealpha=1.0)
    # Nota reubicada arriba a la derecha: la leyenda de dos columnas ocupa
    # la esquina inferior y el cuadro de Spearman ya ocupa la superior
    # izquierda, por lo que este texto solapaba con la leyenda antes.
    ax.text(0.98, 0.98,
             "Nota: efecto techo en Ramsay=6\n(escala no permite valores >6)",
             transform=ax.transAxes, fontsize=7 * ESCALA_FUENTE, color="gray",
             ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8F9FA",
                       edgecolor="#CCCCCC", alpha=0.95))

    fig.tight_layout(rect=[0, 0, 1, 0.88], w_pad=2.8)
    fig.savefig(out("P5_Fig13_Subgrupo_Dosis.png"), facecolor=BG)
    plt.close(fig)
    print("[OK] Fig13 guardada")
    return stats_out


# =============================================================================
# BLOQUE 3C - SUBGRUPO POR PERFIL DE COMPLICACIONES
# =============================================================================

VARS_PERFIL_COMP = [
    "edad_num", "n_sintomas", "dosis_inicial_infusor", "dosis_acum_mid",
    "dosis_primera_ram6", "n_rescates_e2", "tiempo_fallec_dias",
    "ramsay_ultima",
]


def fig14_subgrupo_complicaciones(df):
    """Fig14: heatmap medias + radar comparativo (graficos requeridos 19 y 20)."""
    print("Generando Fig14 - Subgrupo Complicaciones...")
    sub = df.dropna(subset=["evento_adverso"]).copy()
    if len(sub) < 6:
        _fig_vacia("P5_Fig14_Subgrupo_Complicaciones.png",
                    "Fig14: muestra insuficiente (evento adverso no definido)")
        return {}

    g_si = sub[sub["evento_adverso"] == 1]
    g_no = sub[sub["evento_adverso"] == 0]
    if len(g_si) < 2 or len(g_no) < 2:
        _fig_vacia("P5_Fig14_Subgrupo_Complicaciones.png",
                    f"Fig14: grupos demasiado pequenos (con EA={len(g_si)}, sin EA={len(g_no)})")
        return {}

    vars_use = [v for v in VARS_PERFIL_COMP if v in df.columns]
    medias = pd.DataFrame({
        "Sin evento\nadverso": [g_no[v].mean() for v in vars_use],
        "Con evento\nadverso": [g_si[v].mean() for v in vars_use],
    }, index=[LBL_VAR.get(v, v) for v in vars_use])
    # z-score por fila para colorear comparativamente
    z = medias.sub(medias.mean(axis=1), axis=0)
    z = z.div(z.std(axis=1).replace(0, 1), axis=0)

    fig = plt.figure(figsize=(13, max(8, 0.68 * len(vars_use) + 2.6)))
    fig.suptitle(f"FIGURA 14 - Subgrupo: Perfil de Complicaciones — {COHORTE}",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1.0], wspace=0.55)

    # Panel A: heatmap (anotando media real)
    ax_a = fig.add_subplot(gs[0, 0])
    sns.heatmap(z, annot=medias.values, fmt=".1f", cmap="RdBu_r", center=0,
                 cbar_kws={"label": "Z-score (relativo entre grupos)"},
                 linewidths=0.5, linecolor="white",
                 annot_kws={"fontsize": 9 * ESCALA_FUENTE, "fontweight": "bold"}, ax=ax_a)
    ax_a.set_title(f"A · Variables clinicas medias por grupo\n"
                    f"(Sin EA: n={len(g_no)}  ·  Con EA: n={len(g_si)})",
                    fontsize=10 * ESCALA_FUENTE, fontweight="bold")
    ax_a.set_xlabel("Grupo", fontsize=9 * ESCALA_FUENTE)
    ax_a.set_ylabel("Variable", fontsize=9 * ESCALA_FUENTE)
    plt.setp(ax_a.get_xticklabels(), rotation=0)
    plt.setp(ax_a.get_yticklabels(), rotation=0)

    # Panel B: radar chart (min-max normalization global por variable)
    ax_b = fig.add_subplot(gs[0, 1], projection="polar")
    angles = np.linspace(0, 2 * np.pi, len(vars_use), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    def _norm(serie_val, all_vals):
        mn = float(np.nanmin(all_vals)); mx = float(np.nanmax(all_vals))
        if mx - mn < 1e-9: return 0.5
        return (serie_val - mn) / (mx - mn)

    vals_no = []
    vals_si = []
    for v in vars_use:
        all_v = sub[v].dropna().values
        vals_no.append(_norm(float(g_no[v].mean()), all_v))
        vals_si.append(_norm(float(g_si[v].mean()), all_v))
    vals_no_c = vals_no + [vals_no[0]]
    vals_si_c = vals_si + [vals_si[0]]

    ax_b.plot(angles_closed, vals_no_c, "o-", color=PAL_OK[1], lw=2,
               label=f"Sin EA (n={len(g_no)})")
    ax_b.fill(angles_closed, vals_no_c, color=PAL_OK[1], alpha=0.2)
    ax_b.plot(angles_closed, vals_si_c, "o-", color=PAL_ALERTA[1], lw=2,
               label=f"Con EA (n={len(g_si)})")
    ax_b.fill(angles_closed, vals_si_c, color=PAL_ALERTA[1], alpha=0.25)
    ax_b.set_xticks(angles)
    ax_b.set_xticklabels([LBL_VAR.get(v, v) for v in vars_use],
                          fontsize=7.5 * ESCALA_FUENTE)
    ax_b.set_yticks([0.25, 0.5, 0.75]); ax_b.set_yticklabels(["", "", ""])
    ax_b.set_ylim(0, 1)
    ax_b.set_title("B · Perfil clinico comparativo\n(min-max normalizado)",
                    fontsize=10 * ESCALA_FUENTE, fontweight="bold", pad=18)
    ax_b.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=8 * ESCALA_FUENTE)

    # Stats: per-var tests Si vs No
    stats_out = {}
    sig_vars = []
    for v in vars_use:
        tn, st, p = _auto_test_2groups(g_si[v], g_no[v])
        stats_out[f"[Subgrupo] EA {LBL_VAR.get(v, v)}"] = {
            "test": tn,
            "stat": round(float(st), 3) if not pd.isna(st) else np.nan,
            "p":    round(float(p), 4) if not pd.isna(p) else np.nan,
            "sig":  sig_label(p),
        }
        if not pd.isna(p) and p < 0.05:
            sig_vars.append(LBL_VAR.get(v, v))

    fig.tight_layout(rect=[0, 0, 1, 0.90], w_pad=3.0)
    fig.savefig(out("P5_Fig14_Subgrupo_Complicaciones.png"), facecolor=BG)
    plt.close(fig)
    print(f"[OK] Fig14 guardada | grupos: sin EA={len(g_no)}, con EA={len(g_si)}")

    _add_resumen("Subgrupos · Complicaciones",
                  "Variables diferentes en pacientes con evento adverso (p<0.05)",
                  ", ".join(sig_vars) if sig_vars else "Ninguna",
                  np.nan,
                  "Posibles factores de riesgo asociados a complicaciones"
                  if sig_vars else "Sin perfil diferencial detectado")
    return stats_out


# =============================================================================
# BLOQUE 3D - SUBGRUPO POR PAUTA DE ADMINISTRACION
# (Sustituye al subgrupo por dispositivo: TODOS usan infusor.
#  Pauta = uso de bolo inicial de Midazolam (carga) frente a infusion sin bolo)
# =============================================================================

def fig15_subgrupo_pauta(df):
    """Fig15: eficacia por pauta + dosis por pauta (graficos requeridos 21 y 22)."""
    print("Generando Fig15 - Subgrupo Pauta de administracion...")
    sub = df.dropna(subset=["bolo_midazolam"]).copy()
    sub["pauta"] = sub["bolo_midazolam"].map({0: "Sin bolo inicial",
                                                 1: "Con bolo inicial"})
    grupos = ["Sin bolo inicial", "Con bolo inicial"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5))
    fig.suptitle(f"FIGURA 15 - Subgrupo: Pauta de Administracion (con/sin bolo inicial) — {COHORTE}",
                 fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)
    stats_out = {}

    # ── A: eficacia (% sedacion 6) por pauta ────────────────────────────
    ax = axes[0]
    pcts, los, his, ns_si, ns_tot = [], [], [], [], []
    for g in grupos:
        s = sub[sub["pauta"] == g]["sedacion_6"].dropna()
        nsi = int((s == 1).sum()); N = len(s)
        pct, lo, hi = prop_ic95(nsi, N)
        pcts.append(pct); los.append(lo); his.append(hi)
        ns_si.append(nsi); ns_tot.append(N)
    xs = [0, 1]
    cols_b = [PAL_GRAD[1], PAL_GRAD[3]]
    bars = ax.bar(xs, pcts, color=cols_b, alpha=0.85, edgecolor="white",
                   width=0.55)
    ax.errorbar(xs, pcts,
                 yerr=[[p - l for p, l in zip(pcts, los)],
                       [h - p for p, h in zip(pcts, his)]],
                 fmt="none", color=CGLOBAL, capsize=6, lw=1.5)
    for b, p, nsi, N in zip(bars, pcts, ns_si, ns_tot):
        ax.text(b.get_x() + b.get_width() / 2, p + 3,
                f"{p:.1f}%\n(n={nsi}/{N})", ha="center", fontsize=8.5 * ESCALA_FUENTE)
    if all(n > 0 for n in ns_tot):
        ct = np.array([[ns_si[0], ns_tot[0] - ns_si[0]],
                        [ns_si[1], ns_tot[1] - ns_si[1]]])
        try:
            tn, tv, pv = chi2_or_fisher(ct)
        except Exception:
            tn, tv, pv = "—", np.nan, np.nan
    else:
        tn, tv, pv = "—", np.nan, np.nan
    ax.set_xticks(xs); ax.set_xticklabels(grupos, fontsize=9 * ESCALA_FUENTE)
    ax.set_ylabel("% pacientes con sedacion = 6", fontsize=9 * ESCALA_FUENTE)
    ax.set_ylim(0, 125)
    ax.set_title(f"A · Eficacia por pauta\n"
                  f"{tn} ({_fmt_stat(tn, tv)}) p={fmt_p(pv)}",
                  fontsize=10 * ESCALA_FUENTE, fontweight="bold")
    stats_out["[Subgrupo] Sedacion=6 por pauta"] = {
        "test": tn, "stat": tv,
        "p": round(float(pv), 4) if not pd.isna(pv) else np.nan,
        "sig": sig_label(pv),
    }

    # ── B: dosis requerida por pauta ────────────────────────────────────
    ax = axes[1]
    g_no = sub[sub["pauta"] == "Sin bolo inicial"]["dosis_primera_ram6"].dropna()
    g_si = sub[sub["pauta"] == "Con bolo inicial"]["dosis_primera_ram6"].dropna()
    if len(g_no) >= 2 and len(g_si) >= 2:
        bp = ax.boxplot([g_no.values, g_si.values], positions=[1, 2],
                         patch_artist=True, widths=0.45,
                         medianprops=dict(color="white", lw=2),
                         whiskerprops=dict(color=CGLOBAL),
                         capprops=dict(color=CGLOBAL),
                         flierprops=dict(marker="o", markersize=3, alpha=0.5))
        for patch, c in zip(bp["boxes"], cols_b):
            patch.set_facecolor(c); patch.set_alpha(0.8)
        for j, (g, c) in enumerate([(g_no, cols_b[0]), (g_si, cols_b[1])], 1):
            jit = np.random.uniform(-0.13, 0.13, size=len(g))
            ax.scatter(np.full(len(g), j) + jit, g.values, color=c,
                        alpha=0.7, s=45, edgecolors="white", lw=0.5, zorder=3)
            ax.text(j, float(g.max()) * 1.05, f"Med={g.median():.0f}", va="bottom",
                    ha="center", fontsize=8 * ESCALA_FUENTE, color=CGLOBAL, fontweight="bold")
        max_bp = max(float(g_no.max()), float(g_si.max()))
        ax.set_ylim(top=max_bp * 1.28)
        tn2, st2, pv2 = _auto_test_2groups(g_no, g_si)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"Sin bolo\n(n={len(g_no)})",
                              f"Con bolo\n(n={len(g_si)})"], fontsize=9 * ESCALA_FUENTE)
        ax.set_ylabel("Dosis minima eficaz (mg)", fontsize=9 * ESCALA_FUENTE)
        ax.set_title(f"B · Dosis requerida por pauta\n"
                      f"{tn2} p={fmt_p(pv2)}", fontsize=10 * ESCALA_FUENTE, fontweight="bold")
        stats_out["[Subgrupo] Dosis por pauta"] = {
            "test": tn2,
            "stat": round(float(st2), 3) if not pd.isna(st2) else np.nan,
            "p":    round(float(pv2), 4) if not pd.isna(pv2) else np.nan,
            "sig":  sig_label(pv2),
        }
    else:
        ax.text(0.5, 0.5, "Datos insuficientes", ha="center", va="center",
                transform=ax.transAxes, fontsize=10 * ESCALA_FUENTE, color="gray")
        ax.set_title("B · Dosis requerida por pauta",
                      fontsize=10 * ESCALA_FUENTE, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.88], w_pad=2.8)
    fig.savefig(out("P5_Fig15_Subgrupo_Pauta.png"), facecolor=BG)
    plt.close(fig)
    print("[OK] Fig15 guardada")

    _add_resumen("Subgrupos · Pauta",
                  "Eficacia segun bolo inicial",
                  f"Sin bolo={pcts[0]:.0f}% vs Con bolo={pcts[1]:.0f}%",
                  pv,
                  "Diferencia significativa entre pautas"
                  if not pd.isna(pv) and pv < 0.05
                  else "Pautas con eficacia comparable")
    return stats_out


# =============================================================================
# FIG16 - TABLA RESUMEN CONSOLIDADA
# =============================================================================

def fig16_tabla_resumen_consolidada():
    """Fig16: tabla resumen final con todos los hallazgos clave por bloque."""
    import textwrap
    import matplotlib.table
    print("Generando Fig16 - Tabla Resumen Consolidada...")
    if not RESUMEN:
        _fig_vacia("P5_Fig16_Tabla_Resumen_Consolidada.png",
                    "Fig16: sin hallazgos para consolidar")
        return

    cols   = ["Bloque", "Analisis", "Resultado", "p-valor", "Interpretacion"]
    filas_raw = [[r[c] for c in cols] for r in RESUMEN]
    pd.DataFrame(RESUMEN).to_csv(out("P5_tabla_resumen_consolidada.csv"),
                                   index=False, encoding="utf-8-sig")

    # Con la fuente mucho mas grande, ensanchar la figura en la misma
    # proporcion que la fuente anula el efecto visual (la relacion texto/
    # ancho se mantiene igual, asi que al escalarla a un ancho fijo -como
    # \textwidth en LaTeX- se ve igual de grande que antes). Por eso aqui
    # el ancho de figura se mantiene deliberadamente moderado y en cambio
    # se envuelven en varias lineas TODAS las columnas de texto largo
    # (Bloque y Analisis incluidas, no solo Resultado e Interpretacion).
    def _wrap(txt, width):
        s = str(txt)
        return "\n".join(textwrap.wrap(s, width=width)) if s else s

    def _wrap_bloque(txt):
        # Los valores de Bloque son "Palabra" o "Subgrupos · Palabra": se
        # parte en el separador " · " (nunca a mitad de palabra) en vez de
        # envolver por conteo de caracteres.
        s = str(txt)
        return s.replace(" · ", "\n· ", 1)

    filas = []
    line_counts = []
    for row in filas_raw:
        row2 = list(row)
        row2[0] = _wrap_bloque(row2[0])  # Bloque
        row2[1] = _wrap(row2[1], 22)  # Analisis
        row2[2] = _wrap(row2[2], 13)  # Resultado
        row2[4] = _wrap(row2[4], 21)  # Interpretacion
        filas.append(row2)
        line_counts.append(max(row2[0].count("\n") + 1,
                                row2[1].count("\n") + 1,
                                row2[2].count("\n") + 1,
                                row2[4].count("\n") + 1, 1))

    n_rows = len(filas) + 1
    total_lines = sum(line_counts) + 1
    fig_h  = max(11, total_lines * 1.15 + 4.2)
    fig, ax = plt.subplots(figsize=(26, fig_h))
    ax.axis("off")
    fig.suptitle(f"FIGURA 16 - Resumen Consolidado de Hallazgos Estadisticos — {COHORTE}",
                 fontsize=16 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)

    # Menos separacion entre el texto y el borde de cada celda (Cell.PAD por
    # defecto es 0.1): asi el ancho de columna calculado por
    # auto_set_column_width dedica mas espacio al propio texto y menos a
    # margen, dejando margen para una fuente todavia mas grande sin
    # desbordar. Se restaura el valor por defecto al salir de la funcion
    # para no afectar a las demas tablas del script.
    _pad_orig = matplotlib.table.Cell.PAD
    matplotlib.table.Cell.PAD = 0.03
    tbl = ax.table(cellText=filas, colLabels=cols, cellLoc="left",
                    bbox=[0.0, 0.05, 1.0, 0.88])
    tbl.auto_set_font_size(False)

    # Color por bloque (se usa el valor de Bloque SIN envolver, guardado en
    # filas_raw, porque la version envuelta en filas ya lleva saltos de
    # linea y no coincidiria con las claves de este diccionario)
    bloque_color = {
        "ACP":                       "#D6EAF8",
        "Reg. logistica":            "#FCF3CF",
        "Reg. lineal":               "#FAE5D3",
        "Hipotesis":                 "#E8DAEF",
        "Subgrupos · Diagnostico":   "#D4EFDF",
        "Subgrupos · Dosis":         "#D1F2EB",
        "Subgrupos · Complicaciones":"#FADBD8",
        "Subgrupos · Pauta":         "#FDEBD0",
    }
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(CGLOBAL)
            cell.set_text_props(color="white", fontweight="bold", fontsize=23 * ESCALA_FUENTE)
        else:
            bloque = filas_raw[r - 1][0]
            cell.set_facecolor(bloque_color.get(bloque, "white"))
            fs = (21 if c != 0 else 22) * ESCALA_FUENTE
            fw = "bold" if c == 0 else "normal"
            cell.set_text_props(fontsize=fs, fontweight=fw)
        cell.set_edgecolor("#CCCCCC")

    # auto_set_column_width se llama despues de fijar la fuente de cada
    # celda (y tras forzar un draw), para que los anchos se calculen con
    # el tamano de fuente real y no con el valor por defecto de la tabla.
    fig.canvas.draw()
    tbl.auto_set_column_width(col=list(range(len(cols))))
    matplotlib.table.Cell.PAD = _pad_orig

    fig.tight_layout()
    # dpi mas moderado que el savefig.dpi=300 global: esta tabla es muy alta
    # (muchas filas envueltas en varias lineas) y a 300dpi el PNG resultante
    # es innecesariamente pesado sin aportar nitidez extra a este ancho.
    fig.savefig(out("P5_Fig16_Tabla_Resumen_Consolidada.png"), dpi=150, facecolor=BG)
    plt.close(fig)
    print("[OK] Fig16 guardada | CSV: P5_tabla_resumen_consolidada.csv")


# =============================================================================
# MAIN
# =============================================================================

def _run(fn, *a):
    """Ejecuta funcion de figura con captura de errores."""
    try:
        return fn(*a) or {}
    except Exception as e:
        import traceback
        print(f"  [ERROR] {fn.__name__}: {e}")
        traceback.print_exc(limit=2)
        return {}


if __name__ == "__main__":
    np.random.seed(42)

    print(f"\nCargando datos (cohorte: {COHORTE} - filtro CI=1)...")
    df, e2 = load_data()

    stats_all = {}

    print("\n--- BLOQUE 1: ACP ---")
    stats_all.update(_run(fig01_acp_varianza, df))
    stats_all.update(_run(fig02_acp_biplot,    df))
    stats_all.update(_run(fig03_acp_loadings,  df))

    print("\n--- BLOQUE 2A: Regresion logistica ---")
    stats_all.update(_run(fig04_logit_roc,        df))
    stats_all.update(_run(fig05_logit_confusion,  df))
    stats_all.update(_run(fig06_logit_forest,     df))

    print("\n--- BLOQUE 2B: Regresion lineal multiple ---")
    stats_all.update(_run(fig07_lineal_real_vs_pred, df))
    stats_all.update(_run(fig08_lineal_residuos,      df))
    stats_all.update(_run(fig09_lineal_importancia,   df))

    print("\n--- BLOQUE 2C: Pruebas de hipotesis ---")
    stats_all.update(_run(fig10_hipotesis_boxplots, df))
    _run(fig11_hipotesis_tabla, df)

    print("\n--- BLOQUE 3: Subgrupos ---")
    stats_all.update(_run(fig12_subgrupo_diagnostico,    df))
    stats_all.update(_run(fig13_subgrupo_dosis,          df))
    stats_all.update(_run(fig14_subgrupo_complicaciones, df))
    stats_all.update(_run(fig15_subgrupo_pauta,          df))

    print("\n--- Tabla resumen consolidada ---")
    _run(fig16_tabla_resumen_consolidada)

    print("\n" + "=" * 65)
    print(f"  [OK] PASO 5 COMPLETADO ({COHORTE})")
    print("  Bloque 1: ACP (Fig01-03)")
    print("  Bloque 2: Inferencia (Fig04-11)")
    print("  Bloque 3: Subgrupos (Fig12-15)")
    print("  Resumen: Fig16 + 5 CSV exportados")
    print(f"  Salidas en: {OUTPUT_DIR}")
    sig = [k for k, v in stats_all.items()
           if v.get("p") is not np.nan and not pd.isna(v.get("p")) and v["p"] < 0.05]
    print(f"  Hallazgos significativos (p<0.05): {len(sig)}/{len(stats_all)}")
    for s in sig[:8]:
        print(f"    - {s}: p={stats_all[s]['p']}")
    if len(sig) > 8:
        print(f"    ... y {len(sig)-8} mas")
    print("=" * 65)
