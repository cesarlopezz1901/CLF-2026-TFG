"""
=============================================================================
PASO 3 — ANÁLISIS DESCRIPTIVO  [VERSIÓN CON CONSENTIMIENTO INFORMADO]
TFG: Manejo del Paciente Paliativo en Sedación Continua — Hospital La Fe
=============================================================================
FILTRO ACTIVO: Solo se incluyen pacientes con Consentimiento Informado (CI=1).
Genera 7 figuras PNG de alta resolución (180 dpi) + CSV tabla descriptiva.
Paleta: Rojo #C0392B = Caso 1 Oncológico | Azul #1A5276 = Caso 2 No Oncológico

INSTRUCCIONES DE USO:
  pip install pandas openpyxl matplotlib scipy
  python paso3_descriptivo2_consentimiento.py

CONFIGURACIÓN DE RUTAS (modificar si es necesario):
=============================================================================
"""

# ─── RUTAS — ajustar según tu sistema ────────────────────────────────────────
import os

EXCEL_1_PATH = os.environ.get(
    "EXCEL_1", "EXCEL_1_ANONIMIZADO_FINAL.xlsx"
)
EXCEL_2_PATH = os.environ.get(
    "EXCEL_2", "EXCEL_2_ANONIMIZADO_FINAL.xlsx"
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join("P3_ANALISIS_DESCRIPTIVO", "p3_resultados_CI"),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(filename):
    """Ruta completa de archivo de salida."""
    return os.path.join(OUTPUT_DIR, filename)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0. PALETA Y ESTILO GLOBAL
# ──────────────────────────────────────────────────────────────────────────────
C1  = "#C0392B"   # Caso 1 — Oncológico
C2  = "#1A5276"   # Caso 2 — No Oncológico
CGLOBAL = "#2C3E50"
BG  = "#FAFAFA"
GRID_COLOR = "#E0E0E0"

ESCALA_FUENTE = 1.35  # Regla 1: factor global de escalado de fuente para legibilidad en PDF

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   "#CCCCCC",
    "axes.grid":        True,
    "grid.color":       GRID_COLOR,
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "DejaVu Sans",
    "font.size": 9 * ESCALA_FUENTE,
    "axes.titlesize": 10 * ESCALA_FUENTE,
    "axes.titleweight": "bold",
    "axes.labelsize": 9 * ESCALA_FUENTE,
    "xtick.labelsize": 8 * ESCALA_FUENTE,
    "ytick.labelsize": 8 * ESCALA_FUENTE,
    "legend.fontsize": 8 * ESCALA_FUENTE,
    "figure.dpi":       180,
})

LABEL_CASO = {1: "Caso 1 · Oncológico", 2: "Caso 2 · No Oncológico"}
COLOR_CASO = {1: C1, 2: C2}

# ──────────────────────────────────────────────────────────────────────────────
# 1. DICCIONARIOS DE ETIQUETAS CLÍNICAS
# ──────────────────────────────────────────────────────────────────────────────

# Diagnóstico principal (códigos 1-15)
DIAGNOSTICO = {
    1:  "Cáncer cabeza y cuello",
    2:  "Cáncer de mama",
    3:  "Cáncer piel/tej. blandos",
    4:  "Cáncer SNC",
    5:  "Cáncer gastrointestinal",
    6:  "Cáncer genitourinario",
    7:  "Cáncer hematológico",
    8:  "Cáncer tracto respiratorio",
    9:  "Enf. cardiovascular terminal",
    10: "Enf. hepática terminal",
    11: "Enf. infecciosa terminal",
    12: "Enf. neurológica terminal",
    13: "Enf. renal terminal",
    14: "Enf. respiratoria terminal",
    15: "Otros",
}

# Agrupación oncológico vs no oncológico para diagnóstico
DIAGNOSTICO_TIPO = {k: "Oncológico" if k <= 8 else "No oncológico" for k in DIAGNOSTICO}

# Síntoma refractario principal (1-8)
SINTOMA_REFRACTARIO = {
    1: "Dolor",
    2: "Disnea",
    3: "Delirio / Agitación",
    4: "Sufrimiento existencial",
    5: "Convulsiones",
    6: "Náuseas/Vómitos",
    7: "Hemorragia masiva",
    8: "Otros",
}

# Tipo de infusor
INFUSOR = {1: "Elastómero SC", 2: "Elastómero IV", 3: "Presión de gas", 4: "Bomba electrónica IV"}

# Cuidador principal (1-11)
CUIDADOR = {
    1:  "Marido/mujer",
    2:  "Pareja",
    3:  "Hijo/a",
    4:  "Remunerado",
    5:  "Residencia",
    6:  "Padre/madre",
    7:  "Nieto/a",
    8:  "Yerno/nuera",
    9:  "Hermano/a",
    10: "Amigo/a",
    11: "Otro",
}

# Sexo cuidador
SEXO_CUIDADOR = {1: "Mujer", 2: "Hombre", 3: "No binario"}

# Categorías edad cuidador
EDAD_CUIDADOR_BINS   = [0, 50, 65, 75, 85, 150]
EDAD_CUIDADOR_LABELS = ["<50 años", "50-64 años", "65-74 años", "75-84 años", "≥85 años"]

# Vivienda (1=Domicilio, 2=Residencia)
VIVIENDA = {1: "Domicilio", 2: "Residencia"}

# Metástasis
METASTASIS = {1: "Sí", 0: "No", 3: "No aplica"}

# Sexo
SEXO = {1: "Masculino", 2: "Femenino"}

# Escala Ramsay
RAMSAY = {
    3: "3 — Somnoliento,\nresponde a órdenes",
    4: "4 — Dormido,\nrespuesta brusca a luz/sonido",
    6: "6 — Sin respuesta\na estímulos",
}

# Variables binarias síntomas en Excel 1
SINTOMAS_E1 = {
    "Sintomas_Dolor":        "Dolor",
    "Sintomas_Disnea":       "Disnea",
    "Sintomas_Delirio":      "Delirio/Agitación",
    "Sintomas_Sufrimiento":  "Sufrimiento existencial",
    "Sintomas_Convulsiones": "Convulsiones",
    "Sintomas_Nauseas":      "Náuseas",
    "Sintomas_Hemorragia":   "Hemorragia",
    "Sintomas_Otros":        "Otros síntomas",
}

# Fármacos en infusor (Excel 1)
FARMACOS_E1 = {
    "Morfina":          "Morfina",
    "Butilescipolamina":"Butilescopolamina",
    "Metoclopramida":   "Metoclopramida",
    "Haloperidol":      "Haloperidol",
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. CARGA Y LIMPIEZA DE DATOS
# ──────────────────────────────────────────────────────────────────────────────

def _find_col(df, *keywords, require_all=False, exact=False):
    """
    Busca columna que contenga todas (require_all=True) o alguna de las keywords.
    Si exact=True, primero busca coincidencia exacta (strip+lower), luego parcial.
    """
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


def load_data():
    e1 = pd.read_excel(EXCEL_1_PATH)
    e2 = pd.read_excel(EXCEL_2_PATH)

    # Normalizar nombres de columnas (strip espacios, NO forzar minúsculas para no romper referencias)
    e1.columns = e1.columns.str.strip()
    e2.columns = e2.columns.str.strip()

    # ── Edad paciente: buscar columna "Edad" (exacta primero, luego parcial excluyendo cuidador)
    col_edad = _find_col(e1, "edad", exact=True)
    if col_edad is None:
        # Fallback: primera col con "edad" que NO tenga "cuidador"
        col_edad = next(
            (c for c in e1.columns if "edad" in c.lower() and "cuidador" not in c.lower()), None
        )
    if col_edad is None:
        raise ValueError(f"No se encontró columna de edad. Columnas: {list(e1.columns)}")

    raw = e1[col_edad]
    # Usar la API de pandas (no numpy) para detectar tipos — compatible con pandas >= 2.0
    if pd.api.types.is_datetime64_any_dtype(raw):
        # Excel almacenó la edad como fecha por error de formato de celda
        e1["edad_num"] = np.nan
        print(f"   [AVISO] Columna '{col_edad}' es datetime --- no usable como edad")
    elif pd.api.types.is_numeric_dtype(raw):
        e1["edad_num"] = pd.to_numeric(raw, errors="coerce")
    else:
        # object, StringDtype, o cualquier otro tipo texto
        e1["edad_num"] = pd.to_numeric(
            raw.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce"
        )
    print(f"   edad_num: {e1['edad_num'].notna().sum()} válidos de {len(e1)} | "
          f"valores: {e1['edad_num'].dropna().values[:5]}")

    # ── Fechas: buscar columnas de forma robusta
    col_inicio_sed = _find_col(e1, "inicio", "sedaci")
    col_hora_inicio = _find_col(e1, "hora", "inicio", require_all=True)
    col_fallec      = _find_col(e1, "fallecimiento")
    col_hora_fallec = (_find_col(e1, "hora", "xitus", require_all=True)
                        or _find_col(e1, "hora", "fallec", require_all=True))

    def _combinar_fecha_hora(fecha_ser, hora_ser):
        fecha_dt = pd.to_datetime(fecha_ser, errors="coerce", dayfirst=True)
        out = []
        for f, h in zip(fecha_dt, hora_ser if hora_ser is not None else [None]*len(fecha_ser)):
            if pd.isna(f):
                out.append(pd.NaT); continue
            try:
                if pd.notna(h) and isinstance(h, str) and ":" in h:
                    partes = h.strip().split(":")
                    hh = int(partes[0]); mm = int(partes[1]) if len(partes) > 1 else 0
                    f = f.replace(hour=min(hh, 23), minute=min(mm, 59))
                else:
                    f = f.replace(hour=12, minute=0)
            except Exception:
                pass
            out.append(f)
        return pd.Series(out, index=fecha_ser.index)

    if col_inicio_sed and col_fallec:
        ts_inicio = _combinar_fecha_hora(e1[col_inicio_sed],
                                          e1[col_hora_inicio] if col_hora_inicio else None)
        ts_fallec = _combinar_fecha_hora(e1[col_fallec],
                                          e1[col_hora_fallec] if col_hora_fallec else None)
        e1["horas_sed_exitus"] = (ts_fallec - ts_inicio).dt.total_seconds() / 3600
        n_con_hora = int((ts_inicio.dt.hour != 12).sum()) if col_hora_inicio else 0
        print(f"  [CHECK] {n_con_hora}/{len(e1)} pacientes con hora de inicio real detectada")

    # ── Tipo de paciente (buscar columna robustamente)
    col_tipo = _find_col(e1, "tipo", "paciente", require_all=True) or _find_col(e1, "tipo")
    if col_tipo:
        e1["Tipo de paciente"] = e1[col_tipo]
    e1["tipo_label"] = e1["Tipo de paciente"].map(LABEL_CASO)

    # ── Sexo paciente: buscar exactamente "Sexo" o "sexo", nunca "Sexo cuidador"
    col_sexo = next(
        (c for c in e1.columns if c.strip().lower() == "sexo"), None
    ) or next(
        (c for c in e1.columns if "sexo" in c.lower() and "cuidador" not in c.lower()), None
    )
    if col_sexo:
        e1["sexo"] = pd.to_numeric(e1[col_sexo], errors="coerce")
    else:
        e1["sexo"] = np.nan

    # ── Grupos de edad paciente
    bins_e  = [0, 70, 80, 90, 120]
    labs_e  = ["<70 años", "70-79 años", "80-89 años", "≥90 años"]
    e1["grupo_edad"] = pd.cut(e1["edad_num"], bins=bins_e, labels=labs_e, right=False)

    # ── Edad cuidador: buscar columna y categorizar
    col_ec = _find_col(e1, "edad", "cuidador", require_all=True)
    if col_ec:
        raw_ec = e1[col_ec]
        if pd.api.types.is_numeric_dtype(raw_ec):
            e1["edad_cuidador_num"] = pd.to_numeric(raw_ec, errors="coerce")
        else:
            e1["edad_cuidador_num"] = pd.to_numeric(
                raw_ec.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce"
            )
        e1["grupo_edad_cuidador"] = pd.cut(
            e1["edad_cuidador_num"],
            bins=EDAD_CUIDADOR_BINS, labels=EDAD_CUIDADOR_LABELS, right=False,
        )
    else:
        e1["edad_cuidador_num"]  = np.nan
        e1["grupo_edad_cuidador"] = np.nan

    # ── Seguimiento programado (tolera variantes de nombre)
    col_seg = _find_col(e1, "seguimiento")
    if col_seg:
        e1["_seguimiento"] = e1[col_seg]
    else:
        e1["_seguimiento"] = np.nan

    # ── Sexo cuidador
    col_sc = _find_col(e1, "sexo", "cuidador", require_all=True)
    e1["Sexo cuidador_norm"] = e1[col_sc] if col_sc else np.nan

    # ── PAD
    col_pad = _find_col(e1, "pad")
    if col_pad:
        e1["_pad"] = pd.to_numeric(e1[col_pad], errors="coerce")
    else:
        e1["_pad"] = np.nan

    # ── Vivienda
    col_viv = _find_col(e1, "vivienda")
    if col_viv:
        e1["_vivienda"] = pd.to_numeric(e1[col_viv], errors="coerce")
    else:
        e1["_vivienda"] = np.nan

    # ── Cuidador principal
    col_cp = _find_col(e1, "cuidador")
    if col_cp:
        e1["_cuidador"] = pd.to_numeric(e1[col_cp], errors="coerce")
    else:
        e1["_cuidador"] = np.nan

    # ── Dosis numéricas (buscar aproximación en nombre de columna)
    for col in ["Dosis de Midazolam en", "Dosis_Levomeprom",
                "Bolo de Midazolam", "Bolo inicial morfina/butilescopolamina"]:
        if col in e1.columns:
            e1[col] = pd.to_numeric(e1[col], errors="coerce")

    # ── Escala Ramsay (Excel 2)
    col_ramsay = _find_col(e2, "ramsay")
    if col_ramsay:
        e2["Escala Ramsay"] = pd.to_numeric(e2[col_ramsay], errors="coerce")

    # ── Columna Nhc en e2 (buscar robustamente)
    col_nhc_e1 = _find_col(e1, "nhc") or _find_col(e1, "historia") or _find_col(e1, "codigo")
    col_nhc_e2 = _find_col(e2, "nhc") or _find_col(e2, "historia") or _find_col(e2, "codigo")
    if col_nhc_e1 and col_nhc_e2:
        merge_key_e1 = col_nhc_e1
        merge_key_e2 = col_nhc_e2
        if col_nhc_e1 != "Nhc":
            e1["Nhc"] = e1[col_nhc_e1]
        if col_nhc_e2 != "Nhc":
            e2["Nhc"] = e2[col_nhc_e2]

    # ── Merge seguimiento
    e2_tipo = e2.merge(e1[["Nhc", "Tipo de paciente"]], on="Nhc", how="left")

    # Imprimir columnas encontradas para diagnóstico
    print(f"   Col. edad: '{col_edad}' | tipo paciente: '{col_tipo}' | sexo: '{col_sexo}'")
    print(f"   Col. cuidador: '{col_cp}' | vivienda: '{col_viv}' | PAD: '{col_pad}'")
    print(f"   Col. edad cuidador: '{col_ec}' | sexo cuidador: '{col_sc}'")
    print(f"   edad_num: {e1['edad_num'].notna().sum()} válidos | grupo_edad: {e1['grupo_edad'].notna().sum()} válidos")

    # ── FILTRO CONSENTIMIENTO INFORMADO (solo en esta versión) ──────────────
    col_ci = _find_col(e1, "consentimiento")
    if col_ci is None:
        # Buscar variantes: "CI", "consent", "consentim"
        col_ci = next(
            (c for c in e1.columns
             if any(kw in c.lower() for kw in ["consentim", "consent", " ci "])
             or c.strip().lower() == "ci"),
            None,
        )
    if col_ci:
        ci_vals = pd.to_numeric(e1[col_ci], errors="coerce")
        n_total = len(e1)
        e1 = e1[ci_vals == 1].reset_index(drop=True)
        n_ci = len(e1)
        print(f"   [CI] Columna CI: '{col_ci}' | {n_ci} pacientes con CI=1 "
              f"({n_total - n_ci} excluidos por CI negativo/ausente)")
        # Filtrar e2 y recalcular e2_tipo con los NHC que quedan
        nhc_ci = set(e1["Nhc"])
        e2 = e2[e2["Nhc"].isin(nhc_ci)].reset_index(drop=True)
        e2_tipo = e2.merge(e1[["Nhc", "Tipo de paciente"]], on="Nhc", how="left")
    else:
        print("   [CI] AVISO: columna de Consentimiento Informado no encontrada — "
              "se incluyen TODOS los pacientes")
    # ────────────────────────────────────────────────────────────────────────

    # ── PUERTA DE VALIDACIÓN DE DATOS ────────────────────────────────────────
    _n_ci     = len(e1)
    _n_onco   = int((e1["Tipo de paciente"] == 1).sum())
    _n_noonco = int((e1["Tipo de paciente"] == 2).sum())

    _seg_all  = e1["_seguimiento"].dropna()
    _n_seg    = int((_seg_all == 1).sum())
    _seg_g2   = e1.loc[e1["Tipo de paciente"] == 2, "_seguimiento"].dropna()
    _n_seg_g2 = int((_seg_g2 == 1).sum())

    _col_inf  = "Tipo de infusor"
    if _col_inf in e1.columns:
        _inf = pd.to_numeric(e1[_col_inf], errors="coerce")
        _n_sc   = int((_inf == 1).sum())
        _n_iv   = int((_inf == 2).sum())
        _n_bomb = int((_inf == 4).sum())
    else:
        _n_sc = _n_iv = _n_bomb = -1

    _val_err = []
    if _n_ci     != 50: _val_err.append(f"N total CI=1 es {_n_ci}, esperado 50")
    if _n_onco   != 21: _val_err.append(f"Oncologicos es {_n_onco}, esperado 21")
    if _n_noonco != 29: _val_err.append(f"No oncologicos es {_n_noonco}, esperado 29")
    if _n_seg    != 40: _val_err.append(f"Seguimiento programado total es {_n_seg}, esperado 40")
    if _n_seg_g2 != 20: _val_err.append(f"Seguimiento no oncologico es {_n_seg_g2}, esperado 20")
    if _n_sc     != 48: _val_err.append(f"Infusor elastomero SC es {_n_sc}, esperado 48")
    if _n_iv     !=  1: _val_err.append(f"Infusor elastomero IV es {_n_iv}, esperado 1")
    if _n_bomb   !=  1: _val_err.append(f"Bomba electronica IV es {_n_bomb}, esperado 1")

    if _val_err:
        raise RuntimeError(
            "\n[VALIDACION FALLIDA] El Excel anonimizado cargado no coincide con los valores esperados.\n"
            "Comprueba que estas usando EXCEL_1_ANONIMIZADO_FINAL.xlsx correcto.\n"
            "Discrepancias detectadas:\n" + "\n".join(f"  - {e}" for e in _val_err)
        )
    print("   [VALIDACION OK] Cohorte CI=1 coincide con el dataset corregido.")
    # ─────────────────────────────────────────────────────────────────────────

    return e1, e2, e2_tipo


# ──────────────────────────────────────────────────────────────────────────────
# 3. FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────────────────────

def mw_test(g1, g2):
    """Mann-Whitney U, devuelve U y p."""
    g1c = g1.dropna()
    g2c = g2.dropna()
    if len(g1c) < 2 or len(g2c) < 2:
        return np.nan, np.nan
    u, p = mannwhitneyu(g1c, g2c, alternative="two-sided")
    return u, p

def chi2_test(df, var, tipo_col="Tipo de paciente"):
    """Chi2 o Fisher exacto para variable categórica vs tipo de paciente.

    Devuelve (estadistico, p, test_usado) donde test_usado es 'Fisher' o 'Chi2'.
    Usa Fisher exacto cuando la tabla es 2×2 y alguna frecuencia esperada < 5.
    """
    ct = pd.crosstab(df[var], df[tipo_col])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan, np.nan, "Chi2"
    _, expected, _, _ = chi2_contingency(ct)
    if ct.shape == (2, 2) and expected.min() < 5:
        _, p = fisher_exact(ct.values)
        return np.nan, p, "Fisher"
    chi2_stat, p, _, _ = chi2_contingency(ct)
    return chi2_stat, p, "Chi2"

def fmt_p(p):
    """Formatea p-valor para tablas."""
    if pd.isna(p):   return "—"
    if p < 0.001:    return "<0.001"
    if p < 0.05:     return f"{p:.3f}*"
    return f"{p:.3f}"

def cohen_d(g1, g2):
    """Cohen's d para dos grupos."""
    g1c = g1.dropna()
    g2c = g2.dropna()
    if len(g1c) < 2 or len(g2c) < 2:
        return np.nan
    pooled_sd = np.sqrt((g1c.std()**2 + g2c.std()**2) / 2)
    if pooled_sd == 0:
        return np.nan
    return (g1c.mean() - g2c.mean()) / pooled_sd

def med_iqr(series):
    """Devuelve string 'Med [Q1–Q3]'."""
    s = series.dropna()
    if len(s) == 0:
        return "—"
    return f"{s.median():.1f} [{s.quantile(.25):.1f}–{s.quantile(.75):.1f}]"

def pct_n(series):
    """Proporción de valores = 1 sobre no nulos."""
    s = series.dropna()
    if len(s) == 0:
        return "—"
    n = int(s.sum())
    p = n / len(s) * 100
    return f"{n} ({p:.0f}%)"

def _pct_label(threshold=5):
    """Autopct que omite el porcentaje en sectores muy finos (<threshold%)
    para evitar que dos etiquetas contiguas se solapen entre sí; el valor
    exacto sigue disponible en la leyenda."""
    def _fmt(pct):
        return f"{pct:.0f}%" if pct >= threshold else ""
    return _fmt

def title_box(ax, text, color=CGLOBAL):
    ax.set_title(text, color="white", fontsize=9 * ESCALA_FUENTE, fontweight="bold",
                 pad=4, backgroundcolor=color, loc="left")

def add_legend(ax):
    patches = [
        mpatches.Patch(color=C1, label="Caso 1 · Oncológico"),
        mpatches.Patch(color=C2, label="Caso 2 · No Oncológico"),
    ]
    ax.legend(handles=patches, fontsize=7 * ESCALA_FUENTE, framealpha=0.8)


# ──────────────────────────────────────────────────────────────────────────────
# 4. FIGURA 1 — PERFIL GLOBAL DEL PACIENTE
# ──────────────────────────────────────────────────────────────────────────────

def _kpi_card(ax, label, value, color):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.05, 0.08), 0.9, 0.84,
                                boxstyle="round,pad=0.05",
                                facecolor=color, edgecolor="white",
                                linewidth=2, alpha=0.9))
    ax.text(0.5, 0.62, value, ha="center", va="center",
            fontsize=20 * ESCALA_FUENTE, fontweight="bold", color="white")
    ax.text(0.5, 0.24, label, ha="center", va="center",
            fontsize=8 * ESCALA_FUENTE, color="white", multialignment="center")


def _donut(ax, series, mapping, colors, title, center_text="", legend_pos="right"):
    counts = series.map(mapping).value_counts()
    if counts.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9 * ESCALA_FUENTE, fontweight="bold")
        return
    wedges, _, autotexts = ax.pie(
        counts.values,
        labels=None,
        autopct=_pct_label(),
        colors=colors[:len(counts)],
        startangle=90,
        wedgeprops={"width": 0.52, "edgecolor": "white", "linewidth": 2},
        pctdistance=0.76,
    )
    for at in autotexts:
        at.set_fontsize(9 * ESCALA_FUENTE); at.set_fontweight("bold"); at.set_color("white")
    # Paneles estrechos (una sola columna) no tienen sitio a la derecha para
    # la leyenda sin invadir el panel vecino: se coloca debajo en ese caso.
    if legend_pos == "bottom":
        ax.legend(wedges, counts.index, loc="upper center",
                  bbox_to_anchor=(0.5, -0.02), ncol=len(counts),
                  fontsize=7.5 * ESCALA_FUENTE, framealpha=0.9)
    else:
        ax.legend(wedges, counts.index, loc="center left",
                  bbox_to_anchor=(1.0, 0.5), fontsize=7.5 * ESCALA_FUENTE, framealpha=0.9)
    if center_text:
        ax.text(0, 0, center_text, ha="center", va="center",
                fontsize=10 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL)
    ax.set_title(title, fontsize=9 * ESCALA_FUENTE, fontweight="bold", pad=8)


def fig1_perfil_global(e1, e2, e2_tipo):
    n = len(e1)
    n_onco   = int((e1["Tipo de paciente"] == 1).sum())
    n_noonco = int((e1["Tipo de paciente"] == 2).sum())
    edad_med = e1["edad_num"].median()

    fig = plt.figure(figsize=(16, 21))
    fig.suptitle(
        f"FIGURA 1 · PERFIL GLOBAL DEL PACIENTE  (N={n})\n"
        "TFG Sedación Paliativa Continua — Hospital La Fe · Valencia",
        fontsize=13 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL, y=0.99,
    )
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.95, wspace=0.55)

    # ── Fila 0: KPIs
    for i, (lbl, val, col) in enumerate([
        ("Pacientes\ntotales",       str(n),            CGLOBAL),
        ("Oncológicos\nCaso 1",      str(n_onco),       C1),
        ("No Oncológicos\nCaso 2",   str(n_noonco),     C2),
        ("Edad mediana\naños",       f"{edad_med:.0f}", "#7D3C98"),
    ]):
        _kpi_card(fig.add_subplot(gs[0, i]), lbl, val, col)

    # ── Fila 1 col 0-1: Donut Sexo global
    ax_sex = fig.add_subplot(gs[1, :2])
    _donut(ax_sex, e1["sexo"], SEXO,
           ["#5DADE2", "#EC407A"],
           f"Distribución por Sexo  n={n}",
           f"n={n}")

    # ── Fila 1 col 2-3: Donut Tipo de paciente
    ax_tipo = fig.add_subplot(gs[1, 2:])
    _donut(ax_tipo, e1["Tipo de paciente"], LABEL_CASO,
           [C1, C2],
           "Tipo de Paciente",
           f"n={n}")

    # ── Fila 2 col 0-1: Grupos de edad (comparaciones directas, evita problemas Categorical)
    ax_gedad = fig.add_subplot(gs[2, :2])
    edad = e1["edad_num"]
    grupos = ["<70 años", "70-79 años", "80-89 años", "≥90 años"]
    counts_ge = [
        int((edad < 70).sum()),
        int(((edad >= 70) & (edad < 80)).sum()),
        int(((edad >= 80) & (edad < 90)).sum()),
        int((edad >= 90).sum()),
    ]
    colors_ge = ["#AED6F1", "#2E86C1", "#1A5276", "#0B2C50"]
    bars = ax_gedad.bar(grupos, counts_ge, color=colors_ge, edgecolor="white", width=0.6)
    for bar, v in zip(bars, counts_ge):
        if v > 0:
            pct = v / n * 100
            ax_gedad.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                          f"{v}\n({pct:.0f}%)", ha="center", fontsize=8.5 * ESCALA_FUENTE, fontweight="bold")
    ax_gedad.set_ylabel("N.º de pacientes")
    ax_gedad.set_title("Grupos de Edad años", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
    ax_gedad.set_ylim(0, max(counts_ge, default=1) + 2.5)

    # ── Fila 2 col 2: Vivienda (domicilio vs residencia)
    ax_viv = fig.add_subplot(gs[2, 2])
    viv_valid = e1["_vivienda"].dropna()
    if not viv_valid.empty:
        viv_counts = viv_valid.map(VIVIENDA).value_counts()
        ax_viv.bar(range(len(viv_counts)), viv_counts.values,
                   color=["#27AE60", "#E67E22"][:len(viv_counts)], edgecolor="white", width=0.5)
        ax_viv.set_xticks(range(len(viv_counts)))
        ax_viv.set_xticklabels(viv_counts.index, fontsize=8.5 * ESCALA_FUENTE)
        for xi, v in enumerate(viv_counts.values):
            pct = v / n * 100
            ax_viv.text(xi, v + 0.1, f"{v}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=8.5 * ESCALA_FUENTE, fontweight="bold")
        ax_viv.set_ylabel("N.º de pacientes")
        ax_viv.set_ylim(0, viv_counts.values.max() * 1.35)
    else:
        ax_viv.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax_viv.transAxes)
    ax_viv.set_title("Lugar de Residencia", fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    # ── Fila 2 col 3: PAD y Seguimiento programado
    ax_pad = fig.add_subplot(gs[2, 3])
    ind_cols = []
    for col_name, label in [("_pad", "PAD"), ("_seguimiento", "Seguimiento\nprogramado")]:
        s = e1[col_name].dropna()
        if not s.empty:
            ind_cols.append((col_name, label, s.mean() * 100))
    if ind_cols:
        x_pos = range(len(ind_cols))
        vals    = [v for _, _, v in ind_cols]
        xlabels = [l for _, l, _ in ind_cols]
        for xi, (col_name, _, pct) in enumerate(ind_cols):
            ax_pad.bar(xi, pct, color=["#8E44AD", "#16A085"][xi % 2],
                       edgecolor="white", width=0.5)
            ax_pad.text(xi, pct + 1, f"{pct:.0f}%", ha="center", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
        ax_pad.set_xticks(list(x_pos))
        ax_pad.set_xticklabels(xlabels, fontsize=8 * ESCALA_FUENTE)
        ax_pad.set_ylabel("Prevalencia %")
        ax_pad.set_ylim(0, 115)
    else:
        ax_pad.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax_pad.transAxes)
    ax_pad.set_title("PAD y Seguimiento Programado", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
    ax_pad.text(0.5, -0.38,
                "PAD y seguimiento programado son indicadores\nindependientes y no comparables entre sí",
                transform=ax_pad.transAxes, fontsize=7 * ESCALA_FUENTE, color="gray", ha="center")

    # ── Fila 3 col 0-1: Cuidador principal
    ax_cuid = fig.add_subplot(gs[3, :2])
    cp_valid = e1["_cuidador"].dropna()
    if not cp_valid.empty:
        cp_counts = cp_valid.map(CUIDADOR).value_counts()
        colors_cp = plt.cm.Set2.colors
        bars3 = ax_cuid.barh(cp_counts.index, cp_counts.values,
                              color=[colors_cp[i % len(colors_cp)] for i in range(len(cp_counts))],
                              edgecolor="white")
        for bar, v in zip(bars3, cp_counts.values):
            pct = v / n * 100
            ax_cuid.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                         f"{v} ({pct:.0f}%)", va="center", fontsize=7.5 * ESCALA_FUENTE, fontweight="bold")
        ax_cuid.set_xlabel("N.º de pacientes")
        ax_cuid.set_xlim(0, cp_counts.max() + 2)
    else:
        ax_cuid.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax_cuid.transAxes)
    ax_cuid.set_title("Cuidador Principal", fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    # ── Fila 3 col 2: Sexo cuidador
    ax_sc = fig.add_subplot(gs[3, 2])
    if "Sexo cuidador_norm" in e1.columns and e1["Sexo cuidador_norm"].notna().any():
        _donut(ax_sc, e1["Sexo cuidador_norm"], SEXO_CUIDADOR,
               ["#EC407A", "#5DADE2", "#A9A9A9"],
               "Sexo del Cuidador Principal", legend_pos="bottom")
    else:
        ax_sc.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax_sc.transAxes)
        ax_sc.set_title("Sexo del Cuidador Principal", fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    # ── Fila 3 col 3: Edad cuidador por categorías
    ax_ec = fig.add_subplot(gs[3, 3])
    ec_num = e1["edad_cuidador_num"] if "edad_cuidador_num" in e1.columns else pd.Series(dtype=float)
    if ec_num.notna().any():
        ec_counts = [
            int((ec_num < 50).sum()),
            int(((ec_num >= 50) & (ec_num < 65)).sum()),
            int(((ec_num >= 65) & (ec_num < 75)).sum()),
            int(((ec_num >= 75) & (ec_num < 85)).sum()),
            int((ec_num >= 85).sum()),
        ]
        colors_ec = ["#F9E79F", "#F39C12", "#E67E22", "#D35400", "#922B21"]
        barsec = ax_ec.bar(range(len(EDAD_CUIDADOR_LABELS)), ec_counts,
                           color=colors_ec, edgecolor="white", width=0.6)
        for bar, v in zip(barsec, ec_counts):
            if v > 0:
                ax_ec.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                           str(v), ha="center", fontsize=8 * ESCALA_FUENTE, fontweight="bold")
        ax_ec.set_xticks(range(len(EDAD_CUIDADOR_LABELS)))
        ax_ec.set_xticklabels(EDAD_CUIDADOR_LABELS, rotation=30, ha="right", fontsize=7.5 * ESCALA_FUENTE)
        ax_ec.set_ylabel("N.º de pacientes")
    else:
        ax_ec.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax_ec.transAxes)
    ax_ec.set_title("Edad del Cuidador Principal", fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    fig.savefig(out("P3_Fig1_Perfil_Global.png"),
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig1 (Perfil Global) guardada")


# ──────────────────────────────────────────────────────────────────────────────
# 5. FIGURA 2 — COMPARATIVA ONCOLÓGICO vs NO ONCOLÓGICO (Sociodemográfico)
# ──────────────────────────────────────────────────────────────────────────────

def _barras_comparativas(ax, e1, col, mapping, grupos, title, xlabel="N.º de pacientes"):
    col_cp = next((c for c in e1.columns if col.lower() in c.lower()), None)
    if col_cp is None:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9 * ESCALA_FUENTE, fontweight="bold")
        return
    data = (e1.groupby(["Tipo de paciente", col_cp]).size()
              .unstack(fill_value=0).rename(index=LABEL_CASO))
    data.columns = [mapping.get(c, f"Cod {c}") for c in data.columns]
    x_pos = np.arange(len(data))
    colors_stack = plt.cm.Set2.colors
    bottom = np.zeros(len(data))
    for j, col_name in enumerate(data.columns):
        vals = data[col_name].values
        ax.bar(x_pos, vals, bottom=bottom,
               color=colors_stack[j % len(colors_stack)],
               label=col_name, edgecolor="white", alpha=0.88)
        for xi, v, b in zip(x_pos, vals, bottom):
            if v > 0:
                ax.text(xi, b + v/2, str(int(v)), ha="center", va="center",
                        fontsize=8.5 * ESCALA_FUENTE, fontweight="bold", color="white")
        bottom += vals
    ax.set_xticks(x_pos)
    ax.set_xticklabels([t.replace(" · ", "\n") for t in data.index], fontsize=8 * ESCALA_FUENTE)
    ax.set_ylabel(xlabel)
    ax.legend(title=col, fontsize=6.5 * ESCALA_FUENTE, title_fontsize=6.5 * ESCALA_FUENTE, loc="upper right")
    ax.set_title(title, fontsize=9 * ESCALA_FUENTE, fontweight="bold")


def fig2_sociodemografico(e1):
    n1 = int((e1["Tipo de paciente"] == 1).sum())
    n2 = int((e1["Tipo de paciente"] == 2).sum())

    fig = plt.figure(figsize=(13, 19))
    fig.suptitle(
        f"FIGURA 2 · Comparativa Sociodemográfica · Oncológico vs No Oncológico\n"
        f"Caso 1 Oncológico n={n1}  |  Caso 2 No Oncológico n={n2}",
        fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL, y=0.99,
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.95, wspace=0.55)

    g1 = e1[e1["Tipo de paciente"] == 1]
    g2 = e1[e1["Tipo de paciente"] == 2]

    # ── 2A: Boxplot edad
    ax = fig.add_subplot(gs[0, :2])
    bp = ax.boxplot(
        [g1["edad_num"].dropna(), g2["edad_num"].dropna()],
        positions=[1, 2], widths=0.5, patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": CGLOBAL}, capprops={"color": CGLOBAL},
        flierprops={"marker": "o", "color": CGLOBAL, "markersize": 4, "alpha": 0.5},
    )
    for patch, col in zip(bp["boxes"], [C1, C2]):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for grp, col, pos in [(g1, C1, 1), (g2, C2, 2)]:
        jitter = np.random.uniform(-0.15, 0.15, size=len(grp["edad_num"].dropna()))
        ax.scatter(pos + jitter, grp["edad_num"].dropna(), color=col, alpha=0.6, s=25, zorder=3)
        m = grp["edad_num"].median()
        ax.annotate(f"Med={m:.1f}", xy=(pos, m), xytext=(pos + 0.3, m),
                    fontsize=7.5 * ESCALA_FUENTE, color=col, fontweight="bold",
                    arrowprops={"arrowstyle": "->", "color": col, "lw": 1})
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Caso 1\nOncológico", "Caso 2\nNo Oncológico"])
    ax.set_ylabel("Edad años")
    u, p = mw_test(g1["edad_num"], g2["edad_num"])
    d = cohen_d(g1["edad_num"], g2["edad_num"])
    ax.set_title(f"2A · Edad por Tipo de Paciente\nMann-Whitney p={fmt_p(p)} | d={d:.2f}",
                 fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    # ── 2B: Sexo por caso
    ax2 = fig.add_subplot(gs[0, 2:])
    sexo_d = (e1.dropna(subset=["sexo"]).groupby(["Tipo de paciente", "sexo"]).size()
               .unstack(fill_value=0).rename(columns=SEXO).rename(index=LABEL_CASO))
    x_pos = np.arange(len(sexo_d)); w = 0.3
    for j, (col_sex, hatch) in enumerate(zip(sexo_d.columns, ["", "//"])):
        ax2.bar(x_pos + j*w, sexo_d[col_sex], width=w, label=col_sex,
                color=[C1, C2], hatch=hatch, edgecolor="white", alpha=0.85)
        for xi, val in zip(x_pos + j*w, sexo_d[col_sex]):
            ax2.text(xi, val + 0.1, str(int(val)), ha="center", fontsize=8 * ESCALA_FUENTE, fontweight="bold")
    _, p_sex, _ = chi2_test(e1.dropna(subset=["sexo"]), "sexo")
    ax2.set_xticks(x_pos + w/2)
    ax2.set_xticklabels([t.replace(" · ", "\n") for t in sexo_d.index], fontsize=8 * ESCALA_FUENTE)
    ax2.set_ylabel("N.º de pacientes")
    ax2.set_title(f"2B · Sexo por Tipo de Paciente  Chi² p={fmt_p(p_sex)}",
                  fontsize=9 * ESCALA_FUENTE, fontweight="bold")
    ax2.set_ylim(0, sexo_d.values.max() * 1.3)
    ax2.legend(title="Sexo", fontsize=8 * ESCALA_FUENTE, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))

    # ── 2C: Grupos de edad comparativa (comparaciones directas sobre edad_num)
    ax3 = fig.add_subplot(gs[1, :2])
    grupos = ["<70 años", "70-79 años", "80-89 años", "≥90 años"]
    edad_masks = [
        lambda df: df["edad_num"] < 70,
        lambda df: (df["edad_num"] >= 70) & (df["edad_num"] < 80),
        lambda df: (df["edad_num"] >= 80) & (df["edad_num"] < 90),
        lambda df: df["edad_num"] >= 90,
    ]
    x_pos = np.arange(len(grupos))
    for j, (caso, col) in enumerate([(1, C1), (2, C2)]):
        grp = e1[e1["Tipo de paciente"] == caso]
        vals = [int(m(grp).sum()) for m in edad_masks]
        ax3.bar(x_pos + j*0.35, vals, width=0.35,
                label=LABEL_CASO[caso].replace(" · ", "\n"), color=col, edgecolor="white", alpha=0.85)
        for xi, v in zip(x_pos + j*0.35, vals):
            if v > 0:
                ax3.text(xi, v + 0.05, str(v), ha="center", fontsize=8 * ESCALA_FUENTE, fontweight="bold")
    ax3.set_xticks(x_pos + 0.175)
    ax3.set_xticklabels(grupos, fontsize=8 * ESCALA_FUENTE)
    ax3.set_ylabel("N.º de pacientes")
    ax3.set_title("2C · Grupos de Edad por Tipo de Paciente", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
    ax3.set_ylim(0, ax3.get_ylim()[1] * 1.3)
    ax3.legend(fontsize=7 * ESCALA_FUENTE, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))

    # ── 2D: Cuidador principal comparativa (barras agrupadas horizontales)
    ax4 = fig.add_subplot(gs[1, 2:])
    cp_valid = e1["_cuidador"].dropna()
    if not cp_valid.empty:
        tmp = e1[["Tipo de paciente", "_cuidador"]].dropna()
        tmp = tmp.copy()
        tmp["_cuidador_lbl"] = tmp["_cuidador"].map(CUIDADOR)
        g1_cuid = tmp[tmp["Tipo de paciente"] == 1]["_cuidador_lbl"].value_counts()
        g2_cuid = tmp[tmp["Tipo de paciente"] == 2]["_cuidador_lbl"].value_counts()
        all_cats = sorted(set(list(g1_cuid.index) + list(g2_cuid.index)))
        y_pos = np.arange(len(all_cats))
        h = 0.35
        v1 = [int(g1_cuid.get(c, 0)) for c in all_cats]
        v2 = [int(g2_cuid.get(c, 0)) for c in all_cats]
        ax4.barh(y_pos + h / 2, v1, height=h, color=C1, alpha=0.85,
                 label="Caso 1 Oncológico", edgecolor="white")
        ax4.barh(y_pos - h / 2, v2, height=h, color=C2, alpha=0.85,
                 label="Caso 2 No Oncológico", edgecolor="white")
        for yi, v in zip(y_pos + h / 2, v1):
            if v > 0:
                ax4.text(v + 0.05, yi, str(v), va="center", fontsize=7.5 * ESCALA_FUENTE, fontweight="bold")
        for yi, v in zip(y_pos - h / 2, v2):
            if v > 0:
                ax4.text(v + 0.05, yi, str(v), va="center", fontsize=7.5 * ESCALA_FUENTE, fontweight="bold")
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(all_cats, fontsize=8 * ESCALA_FUENTE)
        ax4.set_xlabel("N.º de pacientes")
        ax4.set_xlim(0, max(v1 + v2) * 1.35)
        ax4.legend(fontsize=7 * ESCALA_FUENTE, loc="upper right")
    else:
        ax4.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax4.transAxes)
    ax4.set_title("2D · Cuidador Principal por Tipo de Paciente", fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    # ── 2E: Vivienda comparativa
    ax5 = fig.add_subplot(gs[2, :2])
    viv_valid = e1["_vivienda"].dropna()
    if not viv_valid.empty:
        tmp_v = e1[["Tipo de paciente", "_vivienda"]].dropna()
        viv_d = (tmp_v.groupby(["Tipo de paciente", "_vivienda"]).size()
                  .unstack(fill_value=0).rename(index=LABEL_CASO))
        viv_d.columns = [VIVIENDA.get(int(c) if not pd.isna(c) else c, f"Cod {c}")
                         for c in viv_d.columns]
        x_pos = np.arange(len(viv_d))
        bottom = np.zeros(len(viv_d))
        for j, col_name in enumerate(viv_d.columns):
            vals = viv_d[col_name].values
            ax5.bar(x_pos, vals, bottom=bottom,
                    color=["#27AE60", "#E67E22"][j % 2], label=col_name,
                    edgecolor="white", alpha=0.88)
            for xi, v, b in zip(x_pos, vals, bottom):
                if v > 0:
                    ax5.text(xi, b + v/2, str(int(v)), ha="center", va="center",
                             fontsize=9 * ESCALA_FUENTE, fontweight="bold", color="white")
            bottom += vals
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([t.replace(" · ", "\n") for t in viv_d.index], fontsize=8 * ESCALA_FUENTE)
        ax5.set_ylabel("N.º de pacientes")
        ax5.legend(title="Vivienda", fontsize=7 * ESCALA_FUENTE)
    else:
        ax5.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax5.transAxes)
    ax5.set_title("2E · Domicilio vs Residencia por Tipo de Paciente",
                  fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    # ── 2F: PAD y Seguimiento programado comparativa (sin consentimiento)
    ax6 = fig.add_subplot(gs[2, 2:])
    indicadores = [("_pad", "PAD"), ("_seguimiento", "Seguimiento\nProgramado")]
    ind_disp = [(c, l) for c, l in indicadores if e1[c].notna().any()]
    if ind_disp:
        x_pos = np.arange(len(ind_disp))
        for j, (caso, col) in enumerate([(1, C1), (2, C2)]):
            grp = e1[e1["Tipo de paciente"] == caso]
            pcts = [grp[c].dropna().mean() * 100 for c, _ in ind_disp]
            ax6.bar(x_pos + j*0.35, pcts, width=0.35,
                    label=LABEL_CASO[caso].replace(" · ", "\n"),
                    color=col, edgecolor="white", alpha=0.85)
            for xi, pct in zip(x_pos + j*0.35, pcts):
                ax6.text(xi, pct + 1, f"{pct:.0f}%", ha="center", fontsize=8 * ESCALA_FUENTE, fontweight="bold")
        ax6.set_xticks(x_pos + 0.175)
        ax6.set_xticklabels([l for _, l in ind_disp], fontsize=9 * ESCALA_FUENTE)
        ax6.set_ylabel("Prevalencia %")
        ax6.set_ylim(0, 115)
        ax6.legend(fontsize=7 * ESCALA_FUENTE)
    else:
        ax6.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax6.transAxes)
    ax6.set_title("2F · PAD y Seguimiento Programado por Tipo de Paciente",
                  fontsize=9 * ESCALA_FUENTE, fontweight="bold")
    ax6.text(0.5, -0.32,
             "PAD n=2 en total. Diferencia no interpretable estadísticamente",
             transform=ax6.transAxes, fontsize=7 * ESCALA_FUENTE, color="gray", ha="center")

    fig.savefig(out("P3_Fig2_Comparativa_Sociodemografica.png"),
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig2 (Comparativa Sociodemografica) guardada")


# ──────────────────────────────────────────────────────────────────────────────
# 6. FIGURA 3 — DIAGNÓSTICO Y SÍNTOMA REFRACTARIO
# ──────────────────────────────────────────────────────────────────────────────

def fig3_diagnostico(e1):
    fig = plt.figure(figsize=(16, 15))
    fig.suptitle(
        "FIGURA 3 · Diagnóstico Principal y Síntoma Refractario por Tipo de Paciente",
        fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL, y=0.98,
    )
    # La 3ra columna (3C/3F) lleva barras con etiquetas de categoria largas
    # en el eje Y, por lo que recibe mas ancho relativo que las columnas de
    # quesitos (3A/3B/3D/3E) para que esas etiquetas no invadan el panel vecino.
    # wspace se mantiene moderado: un valor demasiado alto reduce el ancho
    # fisico disponible para cada eje y hace que sus propios titulos de dos
    # lineas se desborden sobre el panel vecino.
    # Columna 2 (indice) actua de separador vacio real entre 3B/3E y 3C/3F:
    # aumentar solo el wspace global no bastaba porque ese hueco es
    # proporcional al ancho de columna, no un ancho fisico propio.
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=1.15, wspace=0.35,
                           width_ratios=[1.0, 1.0, 0.35, 1.2])

    # Abreviaturas locales solo para las etiquetas mostradas en esta figura
    # (no afecta al diccionario global DIAGNOSTICO usado en otras figuras/tablas):
    # con el tamaño de fuente mayor, los nombres largos de diagnostico
    # chocaban entre paneles.
    _ABBR_DIAG_FIG3 = {
        "Enf. cardiovascular terminal": "Enf. cardiovascular t.",
        "Cáncer tracto respiratorio":   "Cáncer tracto resp.",
        "Enf. neurológica terminal":    "Enf. neurológica t.",
        "Enf. infecciosa terminal":     "Enf. infecciosa t.",
    }
    def _abbr_fig3(label):
        return _ABBR_DIAG_FIG3.get(label, label)

    col_diag = "Diagnostico principal"
    col_sint = "Sintoma refractario principal"
    diag_available = col_diag in e1.columns
    sint_available = col_sint in e1.columns

    # ── 3A: Pie diagnóstico Caso 1
    ax1 = fig.add_subplot(gs[0, 0])
    g1 = e1[e1["Tipo de paciente"] == 1]
    n1_total = len(g1)
    if diag_available:
        d1 = g1[col_diag].map(DIAGNOSTICO).value_counts()
        n1_diag = int(g1[col_diag].notna().sum())
        colors1 = plt.cm.Reds(np.linspace(0.4, 0.9, len(d1)))
        wedges, _, autotexts = ax1.pie(
            d1.values, labels=None, autopct=_pct_label(),
            colors=colors1, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(8 * ESCALA_FUENTE); at.set_color("white"); at.set_fontweight("bold")
        ax1.legend([_abbr_fig3(x) for x in d1.index], fontsize=7.5 * ESCALA_FUENTE, loc="lower center",
                   bbox_to_anchor=(0.5, -0.55), ncol=1)
        _title_3a = (f"3A · Diagnóstico Principal\nCaso 1 · Oncológico\nn={n1_diag}"
                     if n1_diag == n1_total
                     else f"3A · Diagnóstico Principal\nCaso 1 · Oncológico\nn registrados={n1_diag} de {n1_total}")
    else:
        _title_3a = f"3A · Diagnóstico Principal\nCaso 1 · Oncológico\nn={n1_total}"
    ax1.set_title(_title_3a, fontsize=10 * ESCALA_FUENTE, fontweight="bold", color=C1)

    # ── 3B: Pie diagnóstico Caso 2
    ax2 = fig.add_subplot(gs[0, 1])
    g2 = e1[e1["Tipo de paciente"] == 2]
    n2_total = len(g2)
    if diag_available:
        d2 = g2[col_diag].map(DIAGNOSTICO).value_counts()
        n2_diag = int(g2[col_diag].notna().sum())
        colors2 = plt.cm.Blues(np.linspace(0.4, 0.9, len(d2)))
        wedges2, _, autotexts2 = ax2.pie(
            d2.values, labels=None, autopct=_pct_label(),
            colors=colors2, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            pctdistance=0.75,
        )
        for at in autotexts2:
            at.set_fontsize(8 * ESCALA_FUENTE); at.set_color("white"); at.set_fontweight("bold")
        ax2.legend([_abbr_fig3(x) for x in d2.index], fontsize=7.5 * ESCALA_FUENTE, loc="lower center",
                   bbox_to_anchor=(0.5, -0.55), ncol=1)
        _title_3b = (f"3B · Diagnóstico Principal\nCaso 2 · No Oncológico\nn={n2_diag}"
                     if n2_diag == n2_total
                     else f"3B · Diagnóstico Principal\nCaso 2 · No Oncológico\nn registrados={n2_diag} de {n2_total}")
    else:
        _title_3b = f"3B · Diagnóstico Principal\nCaso 2 · No Oncológico\nn={n2_total}"
    ax2.set_title(_title_3b, fontsize=10 * ESCALA_FUENTE, fontweight="bold", color=C2)

    # ── 3C: Barras comparativas diagnóstico
    ax3 = fig.add_subplot(gs[0, 3])
    if diag_available:
        all_diags = pd.concat([
            g1[col_diag].map(DIAGNOSTICO),
            g2[col_diag].map(DIAGNOSTICO)
        ]).dropna().unique()
        d1_s = g1[col_diag].map(DIAGNOSTICO).value_counts().reindex(all_diags, fill_value=0)
        d2_s = g2[col_diag].map(DIAGNOSTICO).value_counts().reindex(all_diags, fill_value=0)
        x_pos = np.arange(len(all_diags))
        ax3.barh(x_pos + 0.2, d1_s.values, height=0.35, color=C1, alpha=0.85,
                 label="Caso 1 · Oncológico")
        ax3.barh(x_pos - 0.2, d2_s.values, height=0.35, color=C2, alpha=0.85,
                 label="Caso 2 · No Oncológico")
        ax3.set_yticks(x_pos)
        ax3.set_yticklabels([_abbr_fig3(x) for x in all_diags], fontsize=8.5 * ESCALA_FUENTE)
        ax3.set_xlabel("Número de pacientes", fontsize=9.5 * ESCALA_FUENTE)
        ax3.legend(fontsize=8 * ESCALA_FUENTE)
    ax3.set_title("3C · Comparativa de Diagnósticos\nCaso 1 vs Caso 2", fontsize=10 * ESCALA_FUENTE, fontweight="bold")

    # ── 3D: Síntoma refractario Caso 1
    ax4 = fig.add_subplot(gs[1, 0])
    if sint_available:
        s1 = g1[col_sint].map(SINTOMA_REFRACTARIO).value_counts()
        colors_s1 = plt.cm.Reds(np.linspace(0.4, 0.9, len(s1)))
        wedges, _, autotexts = ax4.pie(
            s1.values, labels=None, autopct=_pct_label(),
            colors=colors_s1, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(8 * ESCALA_FUENTE); at.set_color("white"); at.set_fontweight("bold")
        ax4.legend([_abbr_fig3(x) for x in s1.index], fontsize=8 * ESCALA_FUENTE, loc="lower center",
                   bbox_to_anchor=(0.5, -0.55), ncol=1)
    ax4.set_title("3D · Síntoma Refractario\nPrincipal\nCaso 1 · Oncológico",
                  fontsize=10 * ESCALA_FUENTE, fontweight="bold", color=C1)

    # ── 3E: Síntoma refractario Caso 2
    ax5 = fig.add_subplot(gs[1, 1])
    if sint_available:
        s2 = g2[col_sint].map(SINTOMA_REFRACTARIO).value_counts()
        colors_s2 = plt.cm.Blues(np.linspace(0.4, 0.9, len(s2)))
        wedges2, _, autotexts2 = ax5.pie(
            s2.values, labels=None, autopct=_pct_label(),
            colors=colors_s2, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            pctdistance=0.75,
        )
        for at in autotexts2:
            at.set_fontsize(8 * ESCALA_FUENTE); at.set_color("white"); at.set_fontweight("bold")
        ax5.legend([_abbr_fig3(x) for x in s2.index], fontsize=8 * ESCALA_FUENTE, loc="lower center",
                   bbox_to_anchor=(0.5, -0.55), ncol=1)
    ax5.set_title("3E · Síntoma Refractario\nPrincipal\nCaso 2 · No Oncológico",
                  fontsize=10 * ESCALA_FUENTE, fontweight="bold", color=C2)

    # ── 3F: Barras comparativas síntoma
    ax6 = fig.add_subplot(gs[1, 3])
    if sint_available:
        all_sints = pd.concat([
            g1[col_sint].map(SINTOMA_REFRACTARIO),
            g2[col_sint].map(SINTOMA_REFRACTARIO)
        ]).dropna().unique()
        s1_s = g1[col_sint].map(SINTOMA_REFRACTARIO).value_counts().reindex(all_sints, fill_value=0)
        s2_s = g2[col_sint].map(SINTOMA_REFRACTARIO).value_counts().reindex(all_sints, fill_value=0)
        x_pos = np.arange(len(all_sints))
        ax6.barh(x_pos + 0.2, (s1_s.values / len(g1)) * 100, height=0.35,
                 color=C1, alpha=0.85, label="Caso 1 · Oncológico")
        ax6.barh(x_pos - 0.2, (s2_s.values / len(g2)) * 100, height=0.35,
                 color=C2, alpha=0.85, label="Caso 2 · No Oncológico")
        ax6.set_yticks(x_pos)
        ax6.set_yticklabels([_abbr_fig3(x) for x in all_sints], fontsize=8.5 * ESCALA_FUENTE)
        ax6.set_xlabel("Prevalencia %", fontsize=9.5 * ESCALA_FUENTE)
        ax6.legend(fontsize=8 * ESCALA_FUENTE)
    ax6.set_title("3F · Síntoma Refractario Comparativo\nporcentaje por grupo", fontsize=10 * ESCALA_FUENTE, fontweight="bold")

    fig.text(0.5, 0.01,
             "Los sectores representan pacientes con diagnóstico principal registrado. "
             "n puede ser inferior al total del grupo si hay registros faltantes",
             fontsize=7 * ESCALA_FUENTE, color="gray", ha="center")
    fig.savefig(out("P3_Fig3_Diagnostico.png"),
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig3 guardada")


# ──────────────────────────────────────────────────────────────────────────────
# 3B. FIGURA 3B — SÍNTOMAS "OTROS" (Desglose de síntoma refractario = 8)
# ──────────────────────────────────────────────────────────────────────────────

def fig3b_otros_sintomas(e1):
    """Desglose de síntomas específicos cuando Síntoma refractario = 'Otros' (código 8)."""
    col_sint = "Sintoma refractario principal"
    col_otros = _find_col(e1, "otros", "sintoma", require_all=False)

    if col_sint not in e1.columns or col_otros is None:
        print(f"[SKIP] Fig3B: falta '{col_sint}' o columna de otros síntomas")
        return

    # Filtrar solo registros con síntoma refractario = 8 ('Otros')
    otros_records = e1[pd.to_numeric(e1[col_sint], errors="coerce") == 8].copy()
    if len(otros_records) == 0:
        print("[SKIP] Fig3B: no hay registros con síntoma refractario = 'Otros'")
        return

    # Mapeo de síntomas específicos
    sintomas_map = {
        "shock": "Shock",
        "hipoperfusión": "Hipoperfusión distal",
        "hipoperfusion": "Hipoperfusión distal",
        "vía oral": "Pérdida vía oral",
        "via oral": "Pérdida vía oral",
        "pérdida": "Pérdida vía oral",
        "hipo": "HIPO",
        "mioclonía": "MIOCLONIAS",
        "mioclonias": "MIOCLONIAS",
        "edema": "Edema",
        "ictericia": "Ictericia",
    }

    def parse_sintomas(texto):
        """Parsea texto de otros síntomas y retorna lista de síntomas mapeados."""
        if pd.isna(texto) or not isinstance(texto, str):
            return []

        texto = texto.lower()
        found = []
        for keyword, label in sintomas_map.items():
            if keyword in texto and label not in found:
                found.append(label)
        return found

    # Extraer síntomas por grupo
    g1 = e1[e1["Tipo de paciente"] == 1]
    g2 = e1[e1["Tipo de paciente"] == 2]

    otros_g1 = g1[pd.to_numeric(g1[col_sint], errors="coerce") == 8]
    otros_g2 = g2[pd.to_numeric(g2[col_sint], errors="coerce") == 8]

    sintomas_g1 = []
    sintomas_g2 = []

    for _, row in otros_g1.iterrows():
        sintomas_g1.extend(parse_sintomas(row[col_otros]))
    for _, row in otros_g2.iterrows():
        sintomas_g2.extend(parse_sintomas(row[col_otros]))

    # Contar frecuencia
    from collections import Counter
    count_g1 = Counter(sintomas_g1)
    count_g2 = Counter(sintomas_g2)

    # Todos los síntomas únicos
    all_sintomas = sorted(set(list(count_g1.keys()) + list(count_g2.keys())))
    if not all_sintomas:
        print("[SKIP] Fig3B: no se pudieron extraer síntomas específicos")
        return

    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7.5))
    fig.suptitle(
        "FIGURA 3B · Desglose de Síntomas Específicos (Síntoma Refractario = 'Otros')",
        fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL, y=0.98,
    )

    # Gráfico 1: Caso 1 Oncológico
    vals_g1 = [count_g1.get(s, 0) for s in all_sintomas]
    colors_g1 = plt.cm.Reds(np.linspace(0.4, 0.9, len(all_sintomas)))
    ax1.bar(range(len(all_sintomas)), vals_g1, color=colors_g1, edgecolor="white", alpha=0.85)
    ax1.set_xticks(range(len(all_sintomas)))
    ax1.set_xticklabels(all_sintomas, rotation=45, ha="right", fontsize=8.5 * ESCALA_FUENTE)
    ax1.set_ylabel("Número de pacientes", fontsize=9 * ESCALA_FUENTE)
    ax1.set_title(f"Caso 1 · Oncológico n={len(otros_g1)} pacientes con otros síntomas",
                  fontsize=10 * ESCALA_FUENTE, fontweight="bold", color=C1)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for i, v in enumerate(vals_g1):
        if v > 0:
            ax1.text(i, v + 0.1, str(int(v)), ha="center", fontsize=8 * ESCALA_FUENTE, fontweight="bold")

    # Gráfico 2: Caso 2 No Oncológico
    vals_g2 = [count_g2.get(s, 0) for s in all_sintomas]
    colors_g2 = plt.cm.Blues(np.linspace(0.4, 0.9, len(all_sintomas)))
    ax2.bar(range(len(all_sintomas)), vals_g2, color=colors_g2, edgecolor="white", alpha=0.85)
    ax2.set_xticks(range(len(all_sintomas)))
    ax2.set_xticklabels(all_sintomas, rotation=45, ha="right", fontsize=8.5 * ESCALA_FUENTE)
    ax2.set_ylabel("Número de pacientes", fontsize=9 * ESCALA_FUENTE)
    ax2.set_title(f"Caso 2 · No Oncológico n={len(otros_g2)} pacientes con otros síntomas",
                  fontsize=10 * ESCALA_FUENTE, fontweight="bold", color=C2)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for i, v in enumerate(vals_g2):
        if v > 0:
            ax2.text(i, v + 0.1, str(int(v)), ha="center", fontsize=8 * ESCALA_FUENTE, fontweight="bold")

    fig.tight_layout()
    fig.text(0.5, 0.01,
             "Muestra reducida. Los resultados tienen carácter descriptivo y no permiten inferencia estadística",
             fontsize=7.5 * ESCALA_FUENTE, color="gray", ha="center")
    fig.savefig(out("P3_Fig3B_Otros_Sintomas.png"),
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig3B guardada")


# ──────────────────────────────────────────────────────────────────────────────
# 7. FIGURA 4 — FÁRMACOS Y PAUTAS DE SEDACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def _find_drug_col(e1, *keywords):
    """Busca columna que contenga TODAS las keywords (case-insensitive)."""
    for col in e1.columns:
        cl = col.lower()
        if all(kw.lower() in cl for kw in keywords):
            return col
    return None


def fig4_farmacos(e1):
    fig = plt.figure(figsize=(13, 17))
    fig.suptitle(
        "FIGURA 4 · Fármacos Empleados y Pautas de Sedación\n"
        "Dosis, distribución, fármacos asociados y uso de rescates",
        fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL, y=0.99,
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.9, wspace=0.55)
    g1 = e1[e1["Tipo de paciente"] == 1]
    g2 = e1[e1["Tipo de paciente"] == 2]

    # Localizar columnas clave de forma robusta
    col_mid  = _find_drug_col(e1, "midazolam", "dosis") or _find_drug_col(e1, "midazolam", "infusor")
    col_levo = _find_drug_col(e1, "levomepro") or _find_drug_col(e1, "levomepromazina")
    if col_mid:
        e1[col_mid] = pd.to_numeric(e1[col_mid], errors="coerce")
    if col_levo:
        e1[col_levo] = pd.to_numeric(e1[col_levo], errors="coerce")

    # ── 4A: Boxplot dosis Midazolam
    ax1 = fig.add_subplot(gs[0, :2])
    if col_mid and e1[col_mid].notna().any():
        data_mid = [g1[col_mid].dropna(), g2[col_mid].dropna()]
        bp = ax1.boxplot(data_mid, positions=[1, 2], widths=0.5, patch_artist=True,
                         medianprops={"color": "white", "linewidth": 2.5},
                         whiskerprops={"color": CGLOBAL},
                         capprops={"color": CGLOBAL},
                         flierprops={"marker": "o", "markersize": 5, "alpha": 0.5})
        for patch, col in zip(bp["boxes"], [C1, C2]):
            patch.set_facecolor(col); patch.set_alpha(0.75)
        for grp, col, pos in [(g1, C1, 1), (g2, C2, 2)]:
            jitter = np.random.uniform(-0.12, 0.12, size=len(grp[col_mid].dropna()))
            ax1.scatter(pos + jitter, grp[col_mid].dropna(),
                        color=col, alpha=0.55, s=30, zorder=3)
        u, p = mw_test(g1[col_mid], g2[col_mid])
        d = cohen_d(g1[col_mid], g2[col_mid])
        ax1.set_xticks([1, 2])
        ax1.set_xticklabels(["Caso 1\nOncológico", "Caso 2\nNo Oncológico"])
        ax1.set_ylabel("Dosis de Midazolam mg/24h")
        ax1.set_title(f"4A · Dosis de Midazolam en Infusor mg/24h\n"
                      f"Mann-Whitney p={fmt_p(p)} · Cohen's d={d:.2f}", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
        for grp, col, pos in [(g1, C1, 1), (g2, C2, 2)]:
            m = grp[col_mid].median()
            ax1.annotate(f"Mediana\n{m:.0f} mg", xy=(pos, m), xytext=(pos+0.3, m),
                         fontsize=7.5 * ESCALA_FUENTE, color=col, fontweight="bold",
                         arrowprops={"arrowstyle": "->", "color": col, "lw": 1})

    # ── 4B: Categorías dosis Midazolam (conteo directo, evita bug Categorical)
    ax2 = fig.add_subplot(gs[0, 2:])

    if col_mid and e1[col_mid].notna().any():
        labs_mid = ["Baja\n(<20 mg)", "Media\n(20-39 mg)", "Alta\n(40-59 mg)", "Muy alta\n(≥60 mg)"]
        mid_masks = [
            lambda s: s < 20,
            lambda s: (s >= 20) & (s < 40),
            lambda s: (s >= 40) & (s < 60),
            lambda s: s >= 60,
        ]
        x_pos = np.arange(len(labs_mid))

        for j, (caso, col) in enumerate([(1, C1), (2, C2)]):
            grp_mid = e1.loc[e1["Tipo de paciente"] == caso, col_mid].dropna()
            vals = [int(m(grp_mid).sum()) for m in mid_masks]
            ax2.bar(x_pos + j*0.35, vals, width=0.35,
                    label=LABEL_CASO[caso].replace(" · ", "\n"),
                    color=col, edgecolor="white", alpha=0.85)
            for xi, v in zip(x_pos + j*0.35, vals):
                if v > 0:
                    ax2.text(xi, v + 0.05, str(v), ha="center", fontsize=8 * ESCALA_FUENTE, fontweight="bold")

        ax2.set_xticks(x_pos + 0.175)
        ax2.set_xticklabels(labs_mid, fontsize=8 * ESCALA_FUENTE)
        ax2.set_ylabel("Número de pacientes")
        ax2.set_title(
            "4B · Categorías de Dosis de Midazolam\nestratificación por rango terapéutico",
            fontsize=9 * ESCALA_FUENTE, fontweight="bold"
        )
        ax2.legend(fontsize=7 * ESCALA_FUENTE)
    # ── 4C: Fármacos asociados en infusor (búsqueda case-insensitive)
    ax3 = fig.add_subplot(gs[1, :2])
    farm_keywords = [
        ("morfina",          "Morfina"),
        ("butilesc",         "Butilescopolamina"),
        ("metoclopramida",   "Metoclopramida"),
        ("haloperidol",      "Haloperidol"),
    ]
    farm_disponibles = []
    for kw, label in farm_keywords:
        col_f = next((c for c in e1.columns if kw in c.lower()), None)
        if col_f:
            s = pd.to_numeric(e1[col_f], errors="coerce")
            if s.notna().any():
                farm_disponibles.append((col_f, label, s))
    # Levomepromazina: columna de dosis convertida a binario (recibió ≥1 dosis vs. no)
    if col_levo:
        s_levo = (pd.to_numeric(e1[col_levo], errors="coerce").fillna(0) > 0).astype(float)
        farm_disponibles.append((col_levo, "Levomepromazina", s_levo))
    if farm_disponibles:
        farm_labels = [lbl for _, lbl, _ in farm_disponibles]
        x_pos = np.arange(len(farm_labels))
        for j, (caso, col) in enumerate([(1, C1), (2, C2)]):
            grp_idx = e1["Tipo de paciente"] == caso
            pcts = [s[grp_idx].dropna().mean() * 100 for _, _, s in farm_disponibles]
            ax3.bar(x_pos + j*0.35, pcts, width=0.35,
                    label=LABEL_CASO[caso].replace(" · ", "\n"),
                    color=col, edgecolor="white", alpha=0.85)
            for xi, pct in zip(x_pos + j*0.35, pcts):
                if not np.isnan(pct):
                    # Cuando ambas barras del par estan cerca del techo (>90%)
                    # sus etiquetas quedan casi a la misma altura y se tocan;
                    # se escalona verticalmente por serie para separarlas.
                    y_off = 1 + (j * 10 if pct > 90 else 0)
                    ax3.text(xi, pct + y_off, f"{pct:.0f}%", ha="center", fontsize=8 * ESCALA_FUENTE, fontweight="bold")
        ax3.set_xticks(x_pos + 0.175)
        ax3.set_xticklabels(farm_labels, fontsize=8 * ESCALA_FUENTE, rotation=15, ha="right")
        ax3.set_ylabel("Prevalencia de uso %")
        ax3.set_ylim(0, 125)
        ax3.legend(fontsize=7 * ESCALA_FUENTE)
        # Chi² individual para Haloperidol
        try:
            halo_idx = next((i for i, (_, lbl, _) in enumerate(farm_disponibles)
                             if lbl == "Haloperidol"), None)
            if halo_idx is not None:
                halo_col_name, _, halo_s = farm_disponibles[halo_idx]
                ct_halo = pd.crosstab(e1["Tipo de paciente"],
                                      (halo_s > 0).rename("uso"))
                if ct_halo.shape[1] == 2:
                    _, p_halo, _, _ = chi2_contingency(ct_halo)
                    x_halo = halo_idx + 0.175
                    pct_g1 = halo_s[e1["Tipo de paciente"] == 1].dropna().mean() * 100
                    pct_g2 = halo_s[e1["Tipo de paciente"] == 2].dropna().mean() * 100
                    y_top = max(pct_g1, pct_g2) + 8
                    color_p = C1 if p_halo < 0.05 else "black"
                    ax3.annotate(
                        f"p={fmt_p(p_halo)}",
                        xy=(x_halo, y_top), ha="center", fontsize=8 * ESCALA_FUENTE,
                        color=color_p,
                        fontweight="bold" if p_halo < 0.05 else "normal",
                    )
        except Exception as e_halo:
            print(f"[AVISO] Chi² Haloperidol 4C: {e_halo}")
    ax3.set_title("4C · Fármacos Asociados en el Infusor\nprevalencia de uso por tipo de paciente",
                  fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    # ── 4D: Bolos iniciales — prevalencia (binario 1/0) + dosis mediana si existe
    ax4 = fig.add_subplot(gs[1, 2:])
    # Buscar columnas binarias de bolo (1=sí/0=no) y sus dosis por separado
    bolo_specs = [
        ("bolo",    "midazolam",  "Bolo Midazolam"),
        ("bolo",    "morfina",    "Bolo Morfina"),
        ("bolo",    "butilesc",   "Bolo Butilesc."),
        ("bolo",    "metoclop",   "Bolo Metoclop."),
        ("bolo",    "haloperidol","Bolo Haloperidol"),
    ]
    bolos_encontrados = []
    seen_cols = set()
    for kw1, kw2, label in bolo_specs:
        col_b = next(
            (c for c in e1.columns
             if kw1 in c.lower() and kw2 in c.lower() and c not in seen_cols),
            None
        )
        if col_b:
            s = pd.to_numeric(e1[col_b], errors="coerce")
            if s.notna().any():
                bolos_encontrados.append((col_b, label, s))
                seen_cols.add(col_b)

    # Filtrar solo columnas binarias (max ≤ 1) para el panel 4D
    bolos_binarios = []
    for col_b, lbl, s in bolos_encontrados:
        try:
            vals_all = s.dropna()
            if len(vals_all) > 0 and vals_all.max() <= 1:
                bolos_binarios.append((col_b, lbl, s))
        except Exception:
            pass

    if bolos_binarios:
        blabels = [lbl for _, lbl, _ in bolos_binarios]
        x_pos = np.arange(len(blabels))
        for j, (caso, col) in enumerate([(1, C1), (2, C2)]):
            grp_idx = e1["Tipo de paciente"] == caso
            pcts = [s[grp_idx].dropna().mean() * 100 for _, _, s in bolos_binarios]
            ax4.bar(x_pos + j*0.35, pcts, width=0.35,
                    label=LABEL_CASO[caso].replace(" · ", "\n"),
                    color=col, edgecolor="white", alpha=0.85)
            for xi, pct in zip(x_pos + j*0.35, pcts):
                if not np.isnan(pct) and pct > 0:
                    ax4.text(xi, pct + 1, f"{pct:.0f}%", ha="center", fontsize=8 * ESCALA_FUENTE, fontweight="bold")
        ax4.set_xticks(x_pos + 0.175)
        ax4.set_xticklabels(blabels, fontsize=8 * ESCALA_FUENTE, rotation=15, ha="right")
        ax4.set_ylabel("Pacientes que recibieron bolo pct")
        ax4.set_ylim(0, 115)
        ax4.legend(fontsize=7 * ESCALA_FUENTE)
    elif bolos_encontrados:
        ax4.text(0.5, 0.5, "Sin columnas de bolo binarias disponibles",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=8 * ESCALA_FUENTE, color="gray")
    ax4.set_title("4D · Uso de Bolo Inicial por Tipo de Paciente porcentaje de uso",
                  fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    # ── 4E: Uso de rescates
    ax5 = fig.add_subplot(gs[2, 1:3])
    if "Utilizacion rescate" in e1.columns:
        rescate_data = (
            e1.groupby("Tipo de paciente")["Utilizacion rescate"]
            .agg(["sum", "count"])
            .assign(pct=lambda x: x["sum"] / x["count"] * 100)
            .rename(index=LABEL_CASO)
        )
        colors_r = [C1, C2]
        bars = ax5.bar(range(len(rescate_data)), rescate_data["pct"],
                       color=colors_r, edgecolor="white", width=0.5, alpha=0.85)
        for i, (idx, row) in enumerate(rescate_data.iterrows()):
            ax5.text(i, row["pct"] + 1, f"{row['pct']:.0f}%\n(n={int(row['sum'])})",
                     ha="center", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
        ax5.set_xticks(range(len(rescate_data)))
        ax5.set_xticklabels([t.replace(" · ", "\n") for t in rescate_data.index], fontsize=9 * ESCALA_FUENTE)
        ax5.set_ylabel("Pacientes con rescate %")
        ax5.set_ylim(0, 100)
        # Chi2 test
        ct = pd.crosstab(e1["Tipo de paciente"], e1["Utilizacion rescate"])
        if ct.shape == (2, 2):
            chi2, p, _, _ = chi2_contingency(ct)
            ax5.set_title(f"4E · Utilización de Rescate por Tipo de Paciente\n"
                          f"Chi² p={fmt_p(p)}", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
        else:
            ax5.set_title("4E · Utilización de Rescate por Tipo de Paciente",
                          fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    fig.savefig(out("P3_Fig4_Farmacos.png"),
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig4 guardada")


# ──────────────────────────────────────────────────────────────────────────────
# 8. FIGURA 5 — INFUSORES / DIFUSORES
# ──────────────────────────────────────────────────────────────────────────────

def fig5_infusores(e1, e2, e2_tipo):
    fig = plt.figure(figsize=(17, 14))
    fig.suptitle(
        "FIGURA 5 · Uso y Características de los Sistemas de Infusión (Difusores)\n"
        "Tipo de infusor y eventos durante el seguimiento",
        fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL, y=0.98,
    )
    # 3 paneles arriba (5A-5C) + 2 paneles centrados abajo (5D-5E, antes 5E-5F);
    # se elimina el antiguo panel 5D vacio ("pendiente de datos"). wspace es
    # generoso porque los titulos de dos lineas de 5B/5C/5D/5E son largos y
    # se solapan con el panel vecino si el hueco entre columnas es escaso.
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=1.1, wspace=1.3)
    g1 = e1[e1["Tipo de paciente"] == 1]
    g2 = e1[e1["Tipo de paciente"] == 2]

    # ── 5A: Tipo de infusor global
    ax1 = fig.add_subplot(gs[0, 0:2])
    col_inf = "Tipo de infusor"
    if col_inf in e1.columns:
        inf_counts = e1[col_inf].map(INFUSOR).value_counts()
        colors_inf = ["#E74C3C", "#2E86C1", "#27AE60", "#F39C12"]
        wedges, _, autotexts = ax1.pie(
            inf_counts.values,
            labels=None,
            autopct=_pct_label(),
            colors=colors_inf[:len(inf_counts)],
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            pctdistance=0.72,
        )
        for at in autotexts:
            at.set_fontsize(9 * ESCALA_FUENTE); at.set_fontweight("bold"); at.set_color("white")
        # Leyenda debajo del quesito (no a un lado): el panel 5A comparte
        # fila con 5B/5C y una leyenda lateral invadiria el panel vecino.
        ax1.legend(wedges, inf_counts.index, loc="upper center",
                   bbox_to_anchor=(0.5, -0.02), ncol=1,
                   fontsize=7.5 * ESCALA_FUENTE, framealpha=0.9)

        # Detectar y reportar tipos "otro"
        for code, label in e1[col_inf].items():
            if label not in INFUSOR.values() and pd.notna(label):
                inf_counts_raw = e1[col_inf].value_counts()
                if code in inf_counts_raw.index:
                    print(f"[AVISO] Fig5: Tipo de infusor 'otro' detectado - Código: {code}, Valor: {label}, n={inf_counts_raw[code]}")
    n_inf_global = len(e1)
    ax1.set_title(f"5A · Tipo de Sistema de Infusión\nmuestra global n={n_inf_global}",
                  fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    # ── 5B: Tipo infusor por caso
    ax2 = fig.add_subplot(gs[0, 2:4])
    if col_inf in e1.columns:
        inf_caso = (
            e1.groupby(["Tipo de paciente", col_inf])
            .size()
            .unstack(fill_value=0)
            .rename(index=LABEL_CASO)
        )
        inf_caso.columns = [INFUSOR.get(c, f"Tipo {c}") for c in inf_caso.columns]
        x_pos = np.arange(len(inf_caso))
        bottom = np.zeros(len(inf_caso))
        colors_stack = ["#E74C3C", "#2E86C1", "#27AE60", "#F39C12"]
        for j, col_name in enumerate(inf_caso.columns):
            vals = inf_caso[col_name].values
            ax2.bar(x_pos, vals, bottom=bottom, color=colors_stack[j % len(colors_stack)],
                    label=col_name, edgecolor="white", alpha=0.85)
            for xi, v, b in zip(x_pos, vals, bottom):
                if v > 0:
                    ax2.text(xi, b + v/2, str(int(v)), ha="center", va="center",
                             fontsize=9 * ESCALA_FUENTE, fontweight="bold", color="white")
            bottom += vals
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([t.replace(" · ", "\n") for t in inf_caso.index], fontsize=8.5 * ESCALA_FUENTE)
        ax2.set_ylabel("Número de pacientes")
        # Sin leyenda propia: el tipo de infusor y sus colores ya quedan
        # identificados en el panel 5A.
    ax2.set_title("5B · Tipo de Infusor por Tipo de Paciente\nbarras apiladas",
                  fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    # ── 5C: Tipo infusor en seguimiento (Excel 2)
    ax3 = fig.add_subplot(gs[0, 4:6])
    col_inf2 = "Tipo de infusor"
    if col_inf2 in e2.columns:
        inf_visits = e2_tipo.groupby(["Tipo de paciente", col_inf2]).size().unstack(fill_value=0)
        inf_visits = inf_visits.rename(index=LABEL_CASO)
        inf_visits.columns = [INFUSOR.get(c, f"Tipo {c}") for c in inf_visits.columns]
        x_pos = np.arange(len(inf_visits))
        bottom = np.zeros(len(inf_visits))
        colors_stack = ["#E74C3C", "#2E86C1", "#27AE60", "#F39C12"]
        for j, col_name in enumerate(inf_visits.columns):
            vals = inf_visits[col_name].values
            ax3.bar(x_pos, vals, bottom=bottom, color=colors_stack[j % len(colors_stack)],
                    label=col_name, edgecolor="white", alpha=0.85)
            for xi, v, b in zip(x_pos, vals, bottom):
                if v > 0:
                    ax3.text(xi, b + v/2, str(int(v)), ha="center", va="center",
                             fontsize=9 * ESCALA_FUENTE, fontweight="bold", color="white")
            bottom += vals
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([t.replace(" · ", "\n") for t in inf_visits.index], fontsize=8.5 * ESCALA_FUENTE)
        ax3.set_ylabel("Número de visitas")
        # Sin leyenda propia: el tipo de infusor y sus colores ya quedan
        # identificados en el panel 5A.
    n_visitas = len(e2)
    ax3.set_title(f"5C · Tipo de Infusor en Visitas de Seguimiento\nn={n_visitas} visitas",
                  fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    # ── 5D: Reemplazo de infusor (fila inferior, centrado con 5E)
    ax5 = fig.add_subplot(gs[1, 1:3])
    col_reemplazo = "Reemplazo de infusor"
    if col_reemplazo in e2.columns:
        reemplazo = e2_tipo.groupby("Tipo de paciente")[col_reemplazo].agg(["sum", "count"])
        reemplazo["pct"] = reemplazo["sum"] / reemplazo["count"] * 100
        reemplazo = reemplazo.rename(index=LABEL_CASO)
        ax5.bar(range(len(reemplazo)), reemplazo["pct"], color=[C1, C2],
                edgecolor="white", width=0.5, alpha=0.85)
        _nota_cero = False
        for i, (_, row) in enumerate(reemplazo.iterrows()):
            ax5.text(i, row["pct"] + 1, f"{row['pct']:.0f}%", ha="center",
                     fontsize=10 * ESCALA_FUENTE, fontweight="bold")
            if row["pct"] == 0 and "No Oncológico" in _:
                _nota_cero = True
        # Nota fuera del area de datos (debajo del eje) para no solapar con
        # la etiqueta "0%" de la barra.
        if _nota_cero:
            ax5.text(0.5, -0.42,
                     "El 0% puede reflejar ausencia de registro,\nademás de ausencia de eventos",
                     transform=ax5.transAxes, ha="center", fontsize=7 * ESCALA_FUENTE, color="gray")
        ax5.set_xticks(range(len(reemplazo)))
        ax5.set_xticklabels([t.replace(" · ", "\n") for t in reemplazo.index], fontsize=8.5 * ESCALA_FUENTE)
        ax5.set_ylabel("Visitas con reemplazo %")
        ax5.set_ylim(0, 100)
    ax5.set_title("5D · Reemplazo de Infusor en Seguimiento\nporcentaje de visitas por tipo de paciente",
                  fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    # ── 5E: Control de síntomas en visitas (fila inferior, centrado con 5D)
    ax6 = fig.add_subplot(gs[1, 3:5])
    col_control = "Buen control de sintomas"
    if col_control in e2.columns:
        ctrl = e2_tipo.groupby("Tipo de paciente")[col_control].agg(["sum", "count"])
        ctrl["pct"] = ctrl["sum"] / ctrl["count"] * 100
        ctrl = ctrl.rename(index=LABEL_CASO)
        ax6.bar(range(len(ctrl)), ctrl["pct"], color=[C1, C2],
                edgecolor="white", width=0.5, alpha=0.85)
        for i, (_, row) in enumerate(ctrl.iterrows()):
            ax6.text(i, row["pct"] + 1, f"{row['pct']:.0f}%\nn={int(row['sum'])}",
                     ha="center", fontsize=9 * ESCALA_FUENTE, fontweight="bold")
        ax6.set_xticks(range(len(ctrl)))
        ax6.set_xticklabels([t.replace(" · ", "\n") for t in ctrl.index], fontsize=8.5 * ESCALA_FUENTE)
        ax6.set_ylabel("Visitas con buen control %")
        ax6.set_ylim(0, 110)
    ax6.set_title("5E · Buen Control de Síntomas en Visitas\nporcentaje de visitas por tipo de paciente",
                  fontsize=9 * ESCALA_FUENTE, fontweight="bold")

    fig.savefig(out("P3_Fig5_Infusores.png"),
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig5 guardada")


# ──────────────────────────────────────────────────────────────────────────────
# 5B. TABLA DOSIS FINAL AGRUPADA (desde Excel 2 - última visita enfermería)
# ──────────────────────────────────────────────────────────────────────────────

def fig4b_dosis_final(e1, e2, e2_tipo):
    """Tabla de dosis final agrupada (última visita enfermería de cada paciente)."""
    col_nhc = (_find_col(e2, "nhc") or _find_col(e2, "historia")
               or _find_col(e2, "codigo"))
    col_mid_e2 = _find_drug_col(e2, "midazolam", "dosis") or _find_drug_col(e2, "midazolam")

    if col_nhc is None or col_mid_e2 is None:
        print("[SKIP] Fig4B: falta columna NHC/codigo o dosis Midazolam en Excel 2")
        return

    # Buscar columnas de fecha y hora para identificar última visita
    col_fecha = _find_col(e2, "fecha") or _find_col(e2, "fecha visita")
    col_hora = _find_col(e2, "hora")

    e2_copy = e2.copy()
    e2_copy[col_mid_e2] = pd.to_numeric(e2_copy[col_mid_e2], errors="coerce")

    # Extraer última visita: último día + última hora de ese día
    if col_fecha:
        e2_copy[col_fecha] = pd.to_datetime(e2_copy[col_fecha], errors="coerce")
        if col_hora:
            e2_copy[col_hora] = pd.to_timedelta(e2_copy[col_hora].astype(str), errors="coerce")
            e2_copy["_dt"] = e2_copy[col_fecha] + e2_copy[col_hora]
        else:
            e2_copy["_dt"] = e2_copy[col_fecha]
        valid_idx = e2_copy.groupby(col_nhc)["_dt"].idxmax().dropna()
        if len(valid_idx) > 0:
            e2_last = e2_copy.loc[valid_idx]
        else:
            e2_last = e2_copy.sort_values(col_nhc).drop_duplicates(col_nhc, keep="last")
    else:
        # Si no hay fecha, usar la última fila por paciente
        e2_last = e2_copy.sort_values(col_nhc).drop_duplicates(col_nhc, keep="last")

    # Obtener dosis final
    dosis_final = e2_last[[col_nhc, col_mid_e2]].dropna(subset=[col_mid_e2]).copy()

    # Merge con e1 para obtener tipo de paciente
    dosis_final_tipo = dosis_final.merge(e1[[col_nhc, "Tipo de paciente"]], on=col_nhc, how="left")

    # ── Decisión de presentación: 15 pacientes tienen dosis final exactamente
    # 50 mg, por lo que esta constante afecta de forma notable al reparto
    # entre Media y Alta. Cambia a "alta" para reproducir la figura anterior.
    DOSIS_50_EN = "media"   # "media" -> 50 mg cuenta como Media [30,50]
                             # "alta"  -> 50 mg cuenta como Alta  [>=50]

    if DOSIS_50_EN == "media":
        CAT_BAJA  = "Baja (<30 mg)"
        CAT_MEDIA = "Media (30-50 mg)"
        CAT_ALTA  = "Alta (>50 mg)"
        def _clasif(d):
            if pd.isna(d):   return np.nan
            if d < 30:       return CAT_BAJA
            if d <= 50:      return CAT_MEDIA
            return CAT_ALTA
    else:
        CAT_BAJA  = "Baja (<30 mg)"
        CAT_MEDIA = "Media (30-49 mg)"
        CAT_ALTA  = "Alta (>=50 mg)"
        def _clasif(d):
            if pd.isna(d):   return np.nan
            if d < 30:       return CAT_BAJA
            if d < 50:       return CAT_MEDIA
            return CAT_ALTA

    CATS = [CAT_BAJA, CAT_MEDIA, CAT_ALTA]
    dosis_final_tipo["dosis_grupo"] = dosis_final_tipo[col_mid_e2].apply(_clasif)

    # Calcular para cada tipo de paciente
    rows = []
    for tipo in [1, 2]:
        tipo_label = LABEL_CASO.get(tipo, f"Tipo {tipo}")
        grp = dosis_final_tipo[dosis_final_tipo["Tipo de paciente"] == tipo]
        total = len(grp)

        rows.append([f"{tipo_label}", "", "", ""])

        for cat in CATS:
            n = len(grp[grp["dosis_grupo"] == cat])
            pct = (n / total * 100) if total > 0 else 0
            rows.append(["", cat, str(int(n)), f"{pct:.1f}%"])

    # Renderizar tabla
    headers = ["Tipo de Paciente", "Categoría de Dosis Final", "n", "%"]
    col_widths = [0.25, 0.35, 0.15, 0.15]
    _render_tabla(
        rows, headers, col_widths,
        figsize=(10, max(6, len(rows) * 0.55)),
        title="TABLA · Dosis Final de Midazolam Agrupada\n(Última visita de enfermería - Excel 2)",
        filename="P3_Fig4B_Dosis_Final_Tabla.png"
    )
    print("[OK] Fig4B (Dosis Final) guardada")


# ──────────────────────────────────────────────────────────────────────────────
# 9. FIGURA 6 — HEATMAP CORRELACIONES SPEARMAN
# ──────────────────────────────────────────────────────────────────────────────

def fig6_correlaciones(e1, e2, e2_tipo):
    # Layout en dos filas: arriba un unico panel centrado (Global) con mas
    # ancho que los paneles inferiores; abajo, Caso 1 y Caso 2 lado a lado.
    # Cada panel es una matriz 6x6 con numero en cada celda, por lo que se
    # mantiene un ancho absoluto generoso para que las etiquetas y numeros
    # no se toquen entre celdas contiguas.
    fig = plt.figure(figsize=(18.7, 21.3))
    # Columna 3 (indice) actua de separador vacio real entre Caso 1 y Caso 2:
    # el colorbar de Caso 1 y las etiquetas largas del eje Y de Caso 2 se
    # solapaban cuando solo dependian del wspace (que no reserva ancho fisico
    # propio, solo un hueco proporcional que la propia leyenda desborda).
    gs = gridspec.GridSpec(2, 7, figure=fig, hspace=0.6, wspace=0.55,
                           width_ratios=[1, 1, 1, 0.7, 1, 1, 1])
    ax_global = fig.add_subplot(gs[0, 1:6])
    ax_c1 = fig.add_subplot(gs[1, 0:3])
    ax_c2 = fig.add_subplot(gs[1, 4:7])
    axes = [ax_global, ax_c1, ax_c2]
    fig.suptitle(
        "FIGURA 6 · Correlaciones de Spearman entre Variables Clínicas Clave\n"
        "Global | Caso 1 Oncológico | Caso 2 No Oncológico",
        fontsize=12 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL, y=0.99,
    )

    # Variables para la matriz — buscar columnas de dosis robustamente
    col_mid_c  = _find_drug_col(e1, "midazolam", "dosis") or _find_drug_col(e1, "midazolam", "infusor")
    col_levo_c = _find_drug_col(e1, "levomepro")
    col_bolo_c = _find_drug_col(e1, "bolo", "midazolam")
    col_resc_c = next((c for c in e1.columns if "rescate" in c.lower() or "utilizacion" in c.lower()), None)

    VARS_CORR_RAW = [
        ("edad_num",    "Edad (años)"),
        (col_mid_c,     "Dosis Midazolam\n(mg/24h)"),
        (col_levo_c,    "Dosis Levomepromazina\n(mg/24h)"),
        (col_bolo_c,    "Bolo Midazolam\n(mg)"),
        ("horas_sed_exitus", "Horas Sed.→Éxitus"),
        (col_resc_c,    "Utilización\nRescate"),
    ]
    vars_disp   = [c for c, _ in VARS_CORR_RAW if c and c in e1.columns]
    labels_disp = [lbl for c, lbl in VARS_CORR_RAW if c and c in e1.columns]
    # Abrevia "Dosis" -> "D." para que las etiquetas quepan con la fuente mayor
    labels_disp = [lbl.replace("Dosis ", "D. ", 1) if lbl.startswith("Dosis ") else lbl
                   for lbl in labels_disp]

    datasets = [
        (e1,                          "Global · n=26",       CGLOBAL),
        (e1[e1["Tipo de paciente"]==1], "Caso 1 · Oncológico (n=10)", C1),
        (e1[e1["Tipo de paciente"]==2], "Caso 2 · No Oncológico (n=16)", C2),
    ]

    for ax, (df_, title_, color_) in zip(axes, datasets):
        df_sub = df_[vars_disp].apply(pd.to_numeric, errors="coerce")
        corr_mat = df_sub.corr(method="spearman")
        n_vars = len(corr_mat)

        # Dibuja heatmap manual
        im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_vars))
        ax.set_yticks(range(n_vars))
        ax.set_xticklabels(labels_disp, rotation=45, ha="right", fontsize=10.5 * ESCALA_FUENTE)
        ax.set_yticklabels(labels_disp, fontsize=10.5 * ESCALA_FUENTE)
        if ax is ax_c2:
            # Caso 2 es el panel de la derecha en la fila inferior: sus
            # etiquetas Y a la izquierda invadian el colorbar de Caso 1.
            # Se mueven a la derecha (hay margen libre) y su propio
            # colorbar se coloca a la izquierda para no competir con ellas.
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

        for i in range(n_vars):
            for j in range(n_vars):
                val = corr_mat.values[i, j]
                if not np.isnan(val):
                    txt_color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=10 * ESCALA_FUENTE, color=txt_color, fontweight="bold")

        ax.set_title(f"Correlaciones Spearman\n{title_}",
                     fontsize=11.5 * ESCALA_FUENTE, fontweight="bold", color=color_)

        # Advertencia si subgrupo oncológico (n=10 o menos)
        if "Oncológico" in title_ and "Global" not in title_:
            n_sub = len(df_)
            if n_sub <= 10:
                ax.text(0.5, -0.22,
                        "n insuficiente para inferencia. Solo descriptivo",
                        transform=ax.transAxes, ha="center", fontsize=7 * ESCALA_FUENTE,
                        color="#E74C3C", fontweight="bold")

        # Colorbar individual (Caso 2 lo lleva a la izquierda, ya que sus
        # etiquetas Y ahora estan a la derecha)
        cb_loc = "left" if ax is ax_c2 else "right"
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.06, fraction=0.06,
                     location=cb_loc).set_label("ρ Spearman", fontsize=9.5 * ESCALA_FUENTE)

    fig.savefig(out("P3_Fig6_Correlaciones.png"),
                dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] Fig6 guardada")


# ──────────────────────────────────────────────────────────────────────────────
# 10. FIGURA 7 — TABLA 1 OFICIAL (estadística descriptiva completa)
# ──────────────────────────────────────────────────────────────────────────────

def _render_tabla(rows, headers, col_widths, figsize, title, filename,
                   title_gap=None, cell_fontsize=None):
    """Renderiza una tabla como figura PNG.

    title_gap: si se indica, acerca el título a la cabecera de la tabla
    ampliando el margen superior de los ejes (fig.subplots_adjust) y
    subiendo y_header en esa misma proporción. Si es None se conserva
    el comportamiento por defecto (y_header=0.95, sin ajustar márgenes).
    cell_fontsize: tamaño de fuente de las celdas de datos; si es None
    se usa el tamaño por defecto.
    """
    fig, ax = plt.subplots(figsize=figsize)
    y_header = 0.95
    if title_gap is not None:
        fig.subplots_adjust(top=0.995)
        y_header = 1 - title_gap
    fig.suptitle(title, fontsize=11 * ESCALA_FUENTE, fontweight="bold", color=CGLOBAL, y=0.99)
    ax.axis("off")

    x_starts = [0]
    for w in col_widths[:-1]:
        x_starts.append(x_starts[-1] + w)

    n_rows = len(rows)
    row_h = 0.9 / (n_rows + 1)

    SECTION_MARKER = "__SECTION__"

    for xi, h, w in zip(x_starts, headers, col_widths):
        ax.add_patch(FancyBboxPatch((xi, y_header - row_h), w - 0.005, row_h,
                                    boxstyle="square,pad=0", transform=ax.transAxes,
                                    facecolor=CGLOBAL, edgecolor="white", linewidth=0.5))
        ax.text(xi + w/2, y_header - row_h/2, h, ha="center", va="center",
                transform=ax.transAxes, fontsize=8 * ESCALA_FUENTE, fontweight="bold",
                color="white", multialignment="center")

    for ri, row in enumerate(rows):
        y_top = y_header - row_h * (ri + 1)
        is_section = str(row[0]).startswith(SECTION_MARKER)
        label = str(row[0]).replace(SECTION_MARKER, "")
        display_row = (label,) + tuple(row[1:])
        bg = "#2C3E50" if is_section else ("#F0F4F8" if ri % 2 == 0 else "white")
        txt_c_def = "white" if is_section else "black"

        for ci, (val, xi, w) in enumerate(zip(display_row, x_starts, col_widths)):
            ax.add_patch(FancyBboxPatch((xi, y_top - row_h), w - 0.005, row_h,
                                        boxstyle="square,pad=0", transform=ax.transAxes,
                                        facecolor=bg, edgecolor=GRID_COLOR, linewidth=0.3))
            fw = "bold" if is_section or ci == 0 else "normal"
            col_text = txt_c_def
            if not is_section and ci == len(headers) - 1 and "*" in str(val):
                col_text = "#C0392B"
            cell_fs = cell_fontsize if cell_fontsize is not None else 7.5 * ESCALA_FUENTE
            ax.text(xi + (0.012 if ci == 0 else w/2), y_top - row_h/2, str(val),
                    ha="left" if ci == 0 else "center", va="center",
                    transform=ax.transAxes, fontsize=cell_fs, fontweight=fw, color=col_text)

    ax.text(0, -0.02,
            "* p<0.05 en rojo (Mann-Whitney U cuantitativas; Chi² categóricas). "
            "Med [Q1–Q3]. n (%).",
            transform=ax.transAxes, fontsize=7.5 * ESCALA_FUENTE, color="#555555", style="italic")

    fig.savefig(out(filename), dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def fig7_tabla1(e1, e2, e2_tipo):
    n   = len(e1)
    g1  = e1[e1["Tipo de paciente"] == 1]
    g2  = e1[e1["Tipo de paciente"] == 2]
    n1  = len(g1); n2 = len(g2)
    SM  = "__SECTION__"   # marcador de sección

    # ══════════════════════════════════════════════════════════════
    # TABLA 1a — PERFIL GLOBAL DEL PACIENTE (columna única global)
    # ══════════════════════════════════════════════════════════════
    rows_g = []

    rows_g.append((f"{SM}VARIABLES SOCIODEMOGRÁFICAS", ""))

    # Edad
    rows_g.append(("Edad — mediana [IQR] (años)", med_iqr(e1["edad_num"])))
    for label, mask in [
        ("<70 años",   e1["edad_num"] < 70),
        ("70-79 años", (e1["edad_num"] >= 70) & (e1["edad_num"] < 80)),
        ("80-89 años", (e1["edad_num"] >= 80) & (e1["edad_num"] < 90)),
        ("≥90 años",   e1["edad_num"] >= 90),
    ]:
        rows_g.append((f"  Edad {label}", pct_n(mask.astype(float))))

    # Sexo
    if "sexo" in e1.columns:
        for k, v in SEXO.items():
            rows_g.append((f"  Sexo — {v}", pct_n((e1["sexo"] == k).astype(float))))

    # Tipo de paciente
    rows_g.append((f"{SM}TIPO DE PACIENTE", ""))
    for k, v in LABEL_CASO.items():
        rows_g.append((f"  {v}", pct_n((e1["Tipo de paciente"] == k).astype(float))))

    # Diagnóstico principal
    rows_g.append((f"{SM}DIAGNÓSTICO PRINCIPAL", ""))
    col_diag = next((c for c in e1.columns if "diagnostico" in c.lower() and "principal" in c.lower()), None)
    if col_diag:
        for k, v in DIAGNOSTICO.items():
            cnt = (e1[col_diag] == k).sum()
            if cnt > 0:
                rows_g.append((f"  {v}", pct_n((e1[col_diag] == k).astype(float))))

    # Vivienda (columna normalizada)
    rows_g.append((f"{SM}LUGAR DE RESIDENCIA", ""))
    if e1["_vivienda"].notna().any():
        for k, v in VIVIENDA.items():
            rows_g.append((f"  {v}", pct_n((e1["_vivienda"] == k).astype(float))))

    # Cuidador principal (columna normalizada)
    rows_g.append((f"{SM}CUIDADOR PRINCIPAL", ""))
    col_cp = "_cuidador"
    if e1["_cuidador"].notna().any():
        for k, v in CUIDADOR.items():
            cnt = (e1["_cuidador"] == k).sum()
            if cnt > 0:
                rows_g.append((f"  {v}", pct_n((e1["_cuidador"] == k).astype(float))))

    # Sexo cuidador
    rows_g.append((f"{SM}SEXO DEL CUIDADOR PRINCIPAL", ""))
    if "Sexo cuidador_norm" in e1.columns and e1["Sexo cuidador_norm"].notna().any():
        for k, v in SEXO_CUIDADOR.items():
            rows_g.append((f"  {v}", pct_n((e1["Sexo cuidador_norm"] == k).astype(float))))

    # Edad cuidador
    rows_g.append((f"{SM}EDAD DEL CUIDADOR (categorías)", ""))
    if "grupo_edad_cuidador" in e1.columns and e1["grupo_edad_cuidador"].notna().any():
        for lab in EDAD_CUIDADOR_LABELS:
            rows_g.append((f"  {lab}", pct_n((e1["grupo_edad_cuidador"] == lab).astype(float))))

    # PAD y seguimiento (sin consentimiento)
    rows_g.append((f"{SM}INDICADORES CLÍNICO-ÉTICOS", ""))
    for col_k, label in [("_pad", "PAD (Planif. Anticipada Decisiones)"),
                          ("_seguimiento", "Seguimiento programado")]:
        if e1[col_k].notna().any():
            rows_g.append((f"  {label}", pct_n(e1[col_k].dropna())))
    col_seg_n = "_seguimiento"  # ya normalizada

    headers_g = ["Variable", f"Global  (n={n})"]
    col_widths_g = [0.72, 0.25]

    df_g = pd.DataFrame(rows_g, columns=headers_g)
    df_g["Variable"] = df_g["Variable"].str.replace(SM, "", regex=False)
    df_g.to_csv(out("P3_Tabla1a_Perfil_Global.csv"), index=False, encoding="utf-8-sig")

    _render_tabla(
        rows_g, headers_g, col_widths_g,
        figsize=(11, max(9, len(rows_g) * 0.52)),
        title=(f"TABLA 1a · PERFIL GLOBAL DEL PACIENTE (N={n})\n"
               "n (%) para categóricas | Mediana [IQR] para cuantitativas"),
        filename="P3_Fig7a_Tabla1_Perfil_Global.png",
        title_gap=0.045,
    )
    print("[OK] Fig7a (Perfil Global) guardada")

    # ══════════════════════════════════════════════════════════════
    # TABLA 1b — COMPARATIVA ONCOLÓGICO vs NO ONCOLÓGICO
    # ══════════════════════════════════════════════════════════════
    rows_c = []

    rows_c.append((f"{SM}VARIABLES SOCIODEMOGRÁFICAS", "", "", "", ""))

    # Edad
    _, p_edad = mw_test(g1["edad_num"], g2["edad_num"])
    rows_c.append(("Edad — mediana [IQR] (años)",
                   med_iqr(e1["edad_num"]), med_iqr(g1["edad_num"]),
                   med_iqr(g2["edad_num"]), fmt_p(p_edad)))
    for label, mask_all, mask1, mask2 in [
        ("<70 años",   e1["edad_num"] < 70,   g1["edad_num"] < 70,   g2["edad_num"] < 70),
        ("70-79 años", (e1["edad_num"]>=70)&(e1["edad_num"]<80),
                       (g1["edad_num"]>=70)&(g1["edad_num"]<80),
                       (g2["edad_num"]>=70)&(g2["edad_num"]<80)),
        ("80-89 años", (e1["edad_num"]>=80)&(e1["edad_num"]<90),
                       (g1["edad_num"]>=80)&(g1["edad_num"]<90),
                       (g2["edad_num"]>=80)&(g2["edad_num"]<90)),
        ("≥90 años",   e1["edad_num"]>=90,    g1["edad_num"]>=90,    g2["edad_num"]>=90),
    ]:
        rows_c.append((f"  {label}",
                       pct_n(mask_all.astype(float)),
                       pct_n(mask1.astype(float)),
                       pct_n(mask2.astype(float)), "—"))

    # Sexo
    if e1["sexo"].notna().any():
        _, p_sex, _ = chi2_test(e1.dropna(subset=["sexo"]), "sexo")
        rows_c.append(("Sexo femenino",
                        pct_n((e1["sexo"]==2).astype(float)),
                        pct_n((g1["sexo"]==2).astype(float)),
                        pct_n((g2["sexo"]==2).astype(float)), fmt_p(p_sex)))

    # Diagnóstico
    rows_c.append((f"{SM}DIAGNÓSTICO PRINCIPAL", "", "", "", ""))
    if col_diag:
        for k, v in DIAGNOSTICO.items():
            cnt = (e1[col_diag] == k).sum()
            if cnt > 0:
                rows_c.append((f"  {v}",
                                pct_n((e1[col_diag]==k).astype(float)),
                                pct_n((g1[col_diag]==k).astype(float)),
                                pct_n((g2[col_diag]==k).astype(float)), "—"))

    # Vivienda (usa Fisher si tabla 2×2 con esperado < 5, via chi2_test)
    rows_c.append((f"{SM}LUGAR DE RESIDENCIA", "", "", "", ""))
    if e1["_vivienda"].notna().any():
        _, p_viv, _viv_test_label = chi2_test(e1.dropna(subset=["_vivienda"]), "_vivienda")
        for k, v in VIVIENDA.items():
            rows_c.append((f"  {v}",
                            pct_n((e1["_vivienda"]==k).astype(float)),
                            pct_n((g1["_vivienda"]==k).astype(float)),
                            pct_n((g2["_vivienda"]==k).astype(float)), fmt_p(p_viv) if k==1 else "—"))

    # Cuidador (usando columna normalizada _cuidador)
    if e1["_cuidador"].notna().any():
        _, p_cp, _ = chi2_test(e1.dropna(subset=["_cuidador"]), "_cuidador")
    else:
        p_cp = np.nan
    rows_c.append((f"{SM}CUIDADOR PRINCIPAL", "", "", "", fmt_p(p_cp)))
    if e1["_cuidador"].notna().any():
        for k, v in CUIDADOR.items():
            cnt = (e1["_cuidador"] == k).sum()
            if cnt > 0:
                rows_c.append((f"  {v}",
                                pct_n((e1["_cuidador"]==k).astype(float)),
                                pct_n((g1["_cuidador"]==k).astype(float)),
                                pct_n((g2["_cuidador"]==k).astype(float)), "—"))

    # Sexo cuidador
    if e1["Sexo cuidador_norm"].notna().any():
        tmp_sc = e1.dropna(subset=["Sexo cuidador_norm"])
        _, p_sc, _ = chi2_test(tmp_sc, "Sexo cuidador_norm")
        rows_c.append((f"{SM}SEXO DEL CUIDADOR", "", "", "", fmt_p(p_sc)))
        for k, v in SEXO_CUIDADOR.items():
            rows_c.append((f"  {v}",
                            pct_n((e1["Sexo cuidador_norm"]==k).astype(float)),
                            pct_n((g1["Sexo cuidador_norm"]==k).astype(float)),
                            pct_n((g2["Sexo cuidador_norm"]==k).astype(float)), "—"))
    else:
        rows_c.append((f"{SM}SEXO DEL CUIDADOR", "", "", "", ""))

    # Edad cuidador
    if e1["grupo_edad_cuidador"].notna().any():
        # Convertir Categorical a str para evitar que pd.crosstab incluya categorías vacías
        tmp_ec = e1.dropna(subset=["grupo_edad_cuidador"]).copy()
        tmp_ec["_ec_str"] = tmp_ec["grupo_edad_cuidador"].astype(str)
        _, p_ec, _ = chi2_test(tmp_ec, "_ec_str")
        rows_c.append((f"{SM}EDAD DEL CUIDADOR (categorías)", "", "", "", fmt_p(p_ec)))
        for lab in EDAD_CUIDADOR_LABELS:
            rows_c.append((f"  {lab}",
                            pct_n((e1["grupo_edad_cuidador"]==lab).astype(float)),
                            pct_n((g1["grupo_edad_cuidador"]==lab).astype(float)),
                            pct_n((g2["grupo_edad_cuidador"]==lab).astype(float)), "—"))
    else:
        rows_c.append((f"{SM}EDAD DEL CUIDADOR (categorías)", "", "", "", ""))

    # Indicadores clínico-éticos (sin consentimiento)
    rows_c.append((f"{SM}INDICADORES CLÍNICO-ÉTICOS", "", "", "", ""))
    for col_k, label in [("_pad", "PAD (Planif. Anticipada Decisiones)"),
                          ("_seguimiento", "Seguimiento programado")]:
        if e1[col_k].notna().any():
            _, p_k, _test_k = chi2_test(e1, col_k)
            rows_c.append((f"  {label}",
                            pct_n(e1[col_k].dropna()),
                            pct_n(g1[col_k].dropna()),
                            pct_n(g2[col_k].dropna()), fmt_p(p_k)))

    # Dosis (búsqueda robusta de nombres de columna)
    rows_c.append((f"{SM}VARIABLES CLÍNICAS (FARMACOLÓGICAS)", "", "", "", ""))
    col_mid  = _find_drug_col(e1, "midazolam", "dosis") or _find_drug_col(e1, "midazolam", "infusor")
    col_levo = _find_drug_col(e1, "levomepro")
    col_bolo = _find_drug_col(e1, "bolo", "midazolam")
    if col_mid and e1[col_mid].notna().any():
        _, p_mid = mw_test(g1[col_mid], g2[col_mid])
        rows_c.append(("Dosis Midazolam infusor (mg/24h)",
                        med_iqr(e1[col_mid]), med_iqr(g1[col_mid]),
                        med_iqr(g2[col_mid]), fmt_p(p_mid)))
    if col_levo and e1[col_levo].notna().any():
        _, p_levo = mw_test(g1[col_levo], g2[col_levo])
        rows_c.append(("Dosis Levomepromazina infusor (mg/24h)",
                        med_iqr(e1[col_levo]), med_iqr(g1[col_levo]),
                        med_iqr(g2[col_levo]), fmt_p(p_levo)))
    if col_bolo and e1[col_bolo].notna().any():
        _, p_bolo = mw_test(g1[col_bolo], g2[col_bolo])
        rows_c.append(("Bolo inicial Midazolam (mg)",
                        med_iqr(e1[col_bolo]), med_iqr(g1[col_bolo]),
                        med_iqr(g2[col_bolo]), fmt_p(p_bolo)))
    if "horas_sed_exitus" in e1.columns:
        _, p_h = mw_test(g1["horas_sed_exitus"], g2["horas_sed_exitus"])
        rows_c.append(("Horas sedación → éxitus",
                        med_iqr(e1["horas_sed_exitus"]), med_iqr(g1["horas_sed_exitus"]),
                        med_iqr(g2["horas_sed_exitus"]), fmt_p(p_h)))

    # Escala Ramsay
    rows_c.append((f"{SM}ESCALA RAMSAY (visitas de seguimiento)", "", "", "", ""))
    if "Escala Ramsay" in e2.columns:
        for ram_val, ram_lbl in [(3, "Ramsay 3 (somnoliento)"),
                                  (4, "Ramsay 4 (dormido)"),
                                  (6, "Ramsay 6 (sin respuesta)")]:
            r_all = (e2_tipo["Escala Ramsay"] == ram_val).astype(float)
            r1 = (e2_tipo[e2_tipo["Tipo de paciente"]==1]["Escala Ramsay"] == ram_val).astype(float)
            r2 = (e2_tipo[e2_tipo["Tipo de paciente"]==2]["Escala Ramsay"] == ram_val).astype(float)
            rows_c.append((f"  {ram_lbl}", pct_n(r_all), pct_n(r1), pct_n(r2), "—"))

    headers_c = ["Variable", f"Global\n(n={n})",
                 f"Oncológico\n(n={n1})", f"No Oncológico\n(n={n2})", "p-valor*"]
    col_widths_c = [0.36, 0.14, 0.16, 0.16, 0.10]

    df_c = pd.DataFrame(rows_c, columns=headers_c)
    df_c["Variable"] = df_c["Variable"].str.replace(SM, "", regex=False)
    df_c.to_csv(out("P3_Tabla1b_Comparativa.csv"), index=False, encoding="utf-8-sig")

    _render_tabla(
        rows_c, headers_c, col_widths_c,
        figsize=(16, max(14, len(rows_c) * 0.58)),
        title=(f"TABLA 1b · COMPARATIVA ONCOLÓGICO vs NO ONCOLÓGICO\n"
               "Mediana [IQR] | n pct | *Mann-Whitney U cuantitativas · "
               "Chi2 categoricas · Fisher exacto en tablas 2x2 con frecuencia esperada menor de 5"),
        filename="P3_Fig7b_Tabla1_Comparativa.png",
        title_gap=0.045,
        cell_fontsize=9.5 * ESCALA_FUENTE,
    )
    print("[OK] Fig7b (Comparativa) guardada")
    print("[OK] CSVs Tabla 1a y 1b guardados")


# ──────────────────────────────────────────────────────────────────────────────
# 11. MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  PASO 3 — ANÁLISIS DESCRIPTIVO  ·  Hospital La Fe TFG")
    print("  VERSION: Solo pacientes con Consentimiento Informado (CI=1)")
    print("=" * 65)

    print("\nCargando datos...")
    e1, e2, e2_tipo = load_data()
    print(f"   Excel 1: {len(e1)} pacientes | Columnas: {list(e1.columns[:10])}...")
    print(f"   Excel 2: {len(e2)} visitas   | Columnas: {list(e2.columns[:10])}...")

    print("\nGenerando figuras...")
    fig1_perfil_global(e1, e2, e2_tipo)        # Fig1: Perfil global del paciente
    fig2_sociodemografico(e1)                   # Fig2: Comparativa onco vs no onco
    fig3_diagnostico(e1)                        # Fig3: Diagnóstico y síntoma refractario
    fig3b_otros_sintomas(e1)                    # Fig3B: Desglose de síntomas "OTROS"
    fig4_farmacos(e1)                           # Fig4: Fármacos y pautas
    fig4b_dosis_final(e1, e2, e2_tipo)          # Fig4B: Dosis final agrupada (última visita)
    fig5_infusores(e1, e2, e2_tipo)             # Fig5: Infusores
    fig6_correlaciones(e1, e2, e2_tipo)         # Fig6: Correlaciones Spearman
    fig7_tabla1(e1, e2, e2_tipo)               # Fig7a + Fig7b: Tablas 1a y 1b

    print("\n" + "=" * 65)
    print("  [OK] PASO 3 COMPLETADO")
    print("  Fig1: Perfil Global | Fig2: Comparativa Sociodemografica")
    print("  Fig3: Diagnostico | Fig4: Farmacos | Fig5: Infusores")
    print("  Fig6: Correlaciones | Fig7a: Tabla Global | Fig7b: Tabla Comparativa")
    print(f"  Salidas en: {OUTPUT_DIR}")
    print("=" * 65)

    # ── BLOQUE DE VERIFICACION ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  VERIFICACION DE COHERENCIA — CIFRAS CLAVE")
    print("=" * 65)

    _g1v = e1[e1["Tipo de paciente"] == 1]
    _g2v = e1[e1["Tipo de paciente"] == 2]
    print(f"  Pacientes CI=1 : {len(e1)}  "
          f"(Oncologicos={len(_g1v)}  No oncologicos={len(_g2v)})")

    # Seguimiento programado
    _seg_v   = e1["_seguimiento"].dropna()
    _seg_tot = int((_seg_v == 1).sum())
    _seg_g1  = int((e1.loc[e1["Tipo de paciente"]==1,"_seguimiento"].dropna()==1).sum())
    _seg_g2  = int((e1.loc[e1["Tipo de paciente"]==2,"_seguimiento"].dropna()==1).sum())
    _, _p_seg, _t_seg = chi2_test(e1, "_seguimiento")
    print(f"  Seguimiento programado : total={_seg_tot}  Onco={_seg_g1}  NoOnco={_seg_g2}"
          f"  p={fmt_p(_p_seg)} [{_t_seg}]")

    # Domicilio vs residencia
    if e1["_vivienda"].notna().any():
        _, _p_viv, _t_viv = chi2_test(e1.dropna(subset=["_vivienda"]), "_vivienda")
        _n_dom = int((e1["_vivienda"] == 1).sum())
        _n_res = int((e1["_vivienda"] == 2).sum())
        print(f"  Vivienda : Domicilio={_n_dom}  Residencia={_n_res}"
              f"  p={fmt_p(_p_viv)} [{_t_viv}]")

    # Tipo de infusor (nivel paciente)
    if "Tipo de infusor" in e1.columns:
        _inf_v = pd.to_numeric(e1["Tipo de infusor"], errors="coerce")
        print(f"  Tipo infusor (E1) : SC={int((_inf_v==1).sum())}  "
              f"IV={int((_inf_v==2).sum())}  "
              f"PresGas={int((_inf_v==3).sum())}  "
              f"Bomba={int((_inf_v==4).sum())}")

    # Tabla dosis final — recalculada aqui para verificacion
    _col_nhc_v  = (_find_col(e2, "nhc") or _find_col(e2, "historia")
                   or _find_col(e2, "codigo"))
    _col_mid_v  = _find_drug_col(e2, "midazolam", "dosis") or _find_drug_col(e2, "midazolam")
    _DOSIS_50_EN = "media"   # debe coincidir con la constante en fig4b_dosis_final
    if _col_nhc_v and _col_mid_v:
        _e2c = e2.copy()
        _e2c[_col_mid_v] = pd.to_numeric(_e2c[_col_mid_v], errors="coerce")
        _last = _e2c.sort_values(_col_nhc_v).drop_duplicates(_col_nhc_v, keep="last")
        _df_d = _last[[_col_nhc_v, _col_mid_v]].dropna(subset=[_col_mid_v])
        _df_d = _df_d.merge(e1[[_col_nhc_v, "Tipo de paciente"]], on=_col_nhc_v, how="left")
        if _DOSIS_50_EN == "media":
            def _cv(d): return "Baja" if d<30 else ("Media" if d<=50 else "Alta")
        else:
            def _cv(d): return "Baja" if d<30 else ("Media" if d<50 else "Alta")
        _df_d["cat"] = _df_d[_col_mid_v].apply(_cv)
        print(f"  Dosis final (DOSIS_50_EN='{_DOSIS_50_EN}') :")
        for _tipo, _lbl in [(1,"Onco"),(2,"NoOnco")]:
            _g = _df_d[_df_d["Tipo de paciente"]==_tipo]["cat"].value_counts()
            print(f"    {_lbl}: Baja={_g.get('Baja',0)}  "
                  f"Media={_g.get('Media',0)}  Alta={_g.get('Alta',0)}")
    print("=" * 65)
    # ─────────────────────────────────────────────────────────────────────────
