"""
=============================================================================
PASO 7 — DASHBOARD INTERACTIVO (HTML AUTOCONTENIDO)
Estudio INFUSED-UHD · Hospitalizacion a Domicilio · Hospital Universitario La Fe
=============================================================================
Genera un dashboard HTML autocontenido (Plotly, CDN) con KPIs clinicos sobre
sedacion paliativa continua con infusores. Restringido a pacientes con
Consentimiento Informado = 1.

Fuente de datos: resuelta por utils_io._resolver_paths()
Salida: <OUTPUT_DIR>/P7_Dashboard_Sedacion_Paliativa.html
=============================================================================
"""
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = (os.path.dirname(_SCRIPT_DIR)
         if not os.path.isfile(os.path.join(_SCRIPT_DIR, "utils_io.py"))
         else _SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from utils_io import _resolver_paths, imprimir_configuracion

_E1_PATH, _E2_PATH, _FUENTE, _META = _resolver_paths()
OUTPUT_DIR = os.environ.get("OUTPUT_DIR",
                             os.path.join(_SCRIPT_DIR, "resultados_dashboard"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

imprimir_configuracion("PASO 7 - DASHBOARD INTERACTIVO", _E1_PATH, _E2_PATH,
                        _FUENTE, _META)


def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact, kruskal

import plotly.graph_objects as go
import plotly.io as pio

# =============================================================================
# PALETA Y ESTILO
# =============================================================================
BG_PAGE   = "#F1E6FA"
BG_CARD   = "#FDEAF3"
C_ONCO    = "#E74C3C"
C_NOONCO  = "#2E86DE"
C_OK      = "#2ECC71"
C_WARN    = "#E67E22"
TXT_MAIN  = "#1C2B3A"
TXT_SUB   = "#5D6D7E"
GRID      = "#E8D6EE"
FONT_FAM  = "'Inter', -apple-system, 'Segoe UI', sans-serif"

PLOTLY_LAYOUT_BASE = dict(
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_CARD,
    font=dict(family=FONT_FAM, color=TXT_MAIN, size=12),
    margin=dict(l=60, r=30, t=60, b=40),
)


# =============================================================================
# UTILIDADES (mismo patron que pasos 3-6)
# =============================================================================

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


def _norm_nhc(series):
    return (series.astype(str).str.strip().str.lower()
            .str.replace(r"\.0$", "", regex=True))


def _combine_dt(date_ser, time_ser):
    res = []
    for d, t in zip(date_ser, time_ser):
        if pd.isna(d):
            res.append(pd.NaT)
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
                total_min = round(float(t) * 1440)
                base = base.replace(hour=min(total_min // 60, 23), minute=total_min % 60)
            else:
                base = base.replace(hour=12, minute=0, second=0)
        except Exception as e:
            print(f"  [AVISO] No se pudo parsear hora '{t}' para fecha {d}: {e}")
        res.append(base)
    return pd.Series(res, index=date_ser.index)


def mw_test(g1, g2):
    g1c, g2c = pd.Series(g1).dropna(), pd.Series(g2).dropna()
    if len(g1c) < 2 or len(g2c) < 2:
        return np.nan, np.nan
    u, p = mannwhitneyu(g1c, g2c, alternative="two-sided")
    return u, p


def chi2_or_fisher_p(n1_pos, n1_tot, n2_pos, n2_tot):
    if n1_tot == 0 or n2_tot == 0:
        return np.nan
    ct = np.array([[n1_pos, n1_tot - n1_pos], [n2_pos, n2_tot - n2_pos]])
    try:
        exp = chi2_contingency(ct)[3]
        if (exp < 5).any():
            _, p = fisher_exact(ct)
        else:
            _, p, _, _ = chi2_contingency(ct)
        return p
    except Exception:
        return np.nan


def sig_label(p):
    if pd.isna(p):
        return "ns"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def sig_badge_color(sig):
    return {"ns": TXT_SUB, "*": C_WARN, "**": C_ONCO}.get(sig, TXT_SUB)


def fmt_p(p):
    if pd.isna(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def prop_ic95(n, N):
    if N == 0:
        return 0.0, 0.0, 0.0
    p = n / N
    z = 1.96
    denom = 1 + z ** 2 / N
    center = (p + z ** 2 / (2 * N)) / denom
    half = z * np.sqrt(p * (1 - p) / N + z ** 2 / (4 * N ** 2)) / denom
    return p * 100, max(0, (center - half) * 100), min(100, (center + half) * 100)


# =============================================================================
# CARGA Y CONSTRUCCION DE DATOS (CI=1 obligatorio)
# =============================================================================

LABEL_TIPO = {1: "Oncologico", 2: "No Oncologico"}
COLOR_TIPO = {1: C_ONCO, 2: C_NOONCO}


def load_data():
    """Carga E1+E2, construye dataset a nivel paciente filtrado por CI=1."""
    e1 = pd.read_excel(_E1_PATH); e1.columns = e1.columns.str.strip()
    e2 = pd.read_excel(_E2_PATH); e2.columns = e2.columns.str.strip()

    # ── columnas clave E1 ──────────────────────────────────────────────
    col_nhc_e1   = (_find_col(e1, "nhc") or _find_col(e1, "historia")
                    or _find_col(e1, "codigo"))
    col_tipo     = (_find_col(e1, "tipo", "paciente", require_all=True)
                    or _find_col(e1, "tipo"))
    col_ci       = _find_col(e1, "consentimiento")
    col_fecha_ini = _find_col(e1, "inicio", "sedaci", require_all=True) or _find_col(e1, "inicio")
    col_hora_ini  = _find_col(e1, "hora", "inicio", require_all=True)
    col_fecha_fall = (_find_col(e1, "fallecimiento") or _find_col(e1, "exitus")
                       or _find_col(e1, "muerte"))
    col_hora_fall  = (_find_col(e1, "hora", "xitus", require_all=True)
                       or _find_col(e1, "hora", "fall", require_all=True))
    col_rescate_e1 = (_find_col(e1, "utilizacion", "rescate", require_all=True)
                       or _find_col(e1, "rescate"))

    if not col_nhc_e1:
        raise KeyError("No se encontro columna identificadora (nhc/historia/codigo) en Excel 1")

    e1["Nhc"] = _norm_nhc(e1[col_nhc_e1])
    e1["Tipo de paciente"] = pd.to_numeric(e1[col_tipo], errors="coerce") if col_tipo else np.nan

    if col_fecha_ini:
        e1[col_fecha_ini] = pd.to_datetime(e1[col_fecha_ini], errors="coerce", dayfirst=True)
    if col_fecha_fall:
        e1[col_fecha_fall] = pd.to_datetime(e1[col_fecha_fall], errors="coerce", dayfirst=True)

    e1["ts_inicio_sed"] = _combine_dt(
        e1[col_fecha_ini] if col_fecha_ini else pd.Series([pd.NaT] * len(e1)),
        e1[col_hora_ini] if col_hora_ini else pd.Series([None] * len(e1)))
    e1["ts_fallec"] = _combine_dt(
        e1[col_fecha_fall] if col_fecha_fall else pd.Series([pd.NaT] * len(e1)),
        e1[col_hora_fall] if col_hora_fall else pd.Series([None] * len(e1)))

    e1["tiempo_fallec_dias"] = (
        (e1["ts_fallec"] - e1["ts_inicio_sed"]).dt.total_seconds() / 86400)
    e1.loc[e1["tiempo_fallec_dias"] < 0, "tiempo_fallec_dias"] = np.nan

    e1["uso_rescate_e1"] = (pd.to_numeric(e1[col_rescate_e1], errors="coerce")
                             if col_rescate_e1 else np.nan)

    # ── columnas clave E2 ──────────────────────────────────────────────
    col_nhc_e2  = (_find_col(e2, "nhc") or _find_col(e2, "historia")
                   or _find_col(e2, "codigo"))
    col_ramsay  = _find_col(e2, "ramsay")
    col_fvis    = _find_col(e2, "fecha", "visita", require_all=True) or _find_col(e2, "fecha")
    col_hvis    = _find_col(e2, "hora", "visita", require_all=True) or _find_col(e2, "hora")
    col_mid_e2  = (_find_col(e2, "midazolam", "infusor", require_all=True)
                   or _find_col(e2, "midazolam"))
    col_resc_e2 = (_find_col(e2, "utiliza", "rescate", require_all=True)
                   or _find_col(e2, "rescate"))
    col_compl   = _find_col(e2, "complicaciones")
    col_buenctrl = _find_col(e2, "buen", "control", require_all=True)

    if not col_nhc_e2:
        raise KeyError("No se encontro columna identificadora (nhc/historia/codigo) en Excel 2")

    e2["Nhc"] = _norm_nhc(e2[col_nhc_e2])
    e2["Escala Ramsay"] = pd.to_numeric(e2[col_ramsay], errors="coerce") if col_ramsay else np.nan
    if col_fvis:
        e2[col_fvis] = pd.to_datetime(e2[col_fvis], errors="coerce", dayfirst=True)
    e2["ts_visita"] = _combine_dt(
        e2[col_fvis] if col_fvis else pd.Series([pd.NaT] * len(e2)),
        e2[col_hvis] if col_hvis else pd.Series([None] * len(e2)))
    e2 = e2.sort_values(["Nhc", "ts_visita"]).reset_index(drop=True)

    e2["dosis_mid_visita"] = (pd.to_numeric(e2[col_mid_e2], errors="coerce")
                               if col_mid_e2 else np.nan)
    e2["uso_resc_visita"] = (pd.to_numeric(e2[col_resc_e2], errors="coerce")
                              if col_resc_e2 else np.nan)
    e2["compl_visita"] = (pd.to_numeric(e2[col_compl], errors="coerce")
                          if col_compl else np.nan)
    e2["buenctrl_visita"] = (pd.to_numeric(e2[col_buenctrl], errors="coerce")
                              if col_buenctrl else np.nan)

    # ── agregados por paciente desde E2 ────────────────────────────────
    nhc_e1 = set(e1["Nhc"].dropna())

    def _agg(grp):
        grp = grp.sort_values("ts_visita")
        dosis_acum = grp["dosis_mid_visita"].sum(min_count=1)
        ram_ultima = (grp["Escala Ramsay"].dropna().iloc[-1]
                      if grp["Escala Ramsay"].notna().any() else np.nan)
        any_ram6 = bool((grp["Escala Ramsay"] == 6).any())
        ram6 = grp[grp["Escala Ramsay"] == 6]
        if len(ram6):
            ts_pram6 = ram6["ts_visita"].iloc[0]
            dosis_pram6 = ram6["dosis_mid_visita"].iloc[0]
        else:
            ts_pram6, dosis_pram6 = pd.NaT, np.nan
        n_resc = int(grp["uso_resc_visita"].fillna(0).gt(0).sum())
        ea_compl = int((grp["compl_visita"].fillna(0) > 0).any())
        buen_ctrl_pct = (grp["buenctrl_visita"].dropna().eq(1).mean() * 100
                         if grp["buenctrl_visita"].notna().any() else np.nan)
        return pd.Series({
            "dosis_acum_mid": float(dosis_acum) if not pd.isna(dosis_acum) else np.nan,
            "ramsay_ultima": float(ram_ultima) if not pd.isna(ram_ultima) else np.nan,
            "any_ramsay6": any_ram6,
            "ts_primera_ram6": ts_pram6,
            "dosis_primera_ram6": float(dosis_pram6) if not pd.isna(dosis_pram6) else np.nan,
            "n_rescates_e2": n_resc,
            "ea_complicacion": ea_compl,
            "buen_ctrl_pct": buen_ctrl_pct,
            "n_visitas": len(grp),
        })

    agg = (e2[e2["Nhc"].isin(nhc_e1)].groupby("Nhc").apply(_agg).reset_index()
           if len(e2) else pd.DataFrame())

    df = e1.merge(agg, on="Nhc", how="inner")

    n_con_hora = int((df["ts_inicio_sed"].dt.hour != 0).sum())
    print(f"  [CHECK] {n_con_hora}/{len(df)} pacientes con hora de inicio de sedacion distinta de medianoche")

    df["tiempo_hasta_ram6_h"] = (
        (df["ts_primera_ram6"] - df["ts_inicio_sed"]).dt.total_seconds() / 3600)
    df.loc[df["tiempo_hasta_ram6_h"] < 0, "tiempo_hasta_ram6_h"] = np.nan

    df["uso_rescate_any"] = (
        (pd.to_numeric(df["uso_rescate_e1"], errors="coerce").fillna(0) > 0)
        | (df["n_rescates_e2"].fillna(0) > 0)).astype(int)

    # ── filtro obligatorio: CI = 1 ──────────────────────────────────────
    if col_ci:
        ci_vals = pd.to_numeric(e1.set_index("Nhc")[col_ci], errors="coerce")
        ci_map = ci_vals.to_dict()
        df["ci"] = df["Nhc"].map(ci_map)
        n_pre = len(df)
        df = df[df["ci"] == 1].reset_index(drop=True)
        print(f"  [CI] {len(df)} pacientes con CI=1 ({n_pre - len(df)} excluidos)")
    else:
        warnings.warn("No se encontro columna de Consentimiento Informado; "
                       "no se aplico filtro CI=1.")

    df["tipo_label"] = df["Tipo de paciente"].map(LABEL_TIPO)
    print(f"\n=== Dataset Paso 7: {len(df)} pacientes (CI=1) ===")
    print(df["tipo_label"].value_counts(dropna=False))

    return df, e2


# =============================================================================
# CALCULO DE KPIs
# =============================================================================

def _split(df, col):
    onco = df.loc[df["Tipo de paciente"] == 1, col]
    noon = df.loc[df["Tipo de paciente"] == 2, col]
    glob = df[col]
    return glob, onco, noon


def compute_kpis(df):
    """Devuelve dict con los 7 KPIs (global / onco / no-onco) + metadatos de test."""
    kpis = {}
    n_total = len(df)
    n_onco = int((df["Tipo de paciente"] == 1).sum())
    n_noonco = int((df["Tipo de paciente"] == 2).sum())

    # KPI 1 — % Ramsay 6 alcanzado
    g, o, na = _split(df, "any_ramsay6")
    p_glob, *_ = prop_ic95(int(g.sum()), len(g))
    p_o, *_ = prop_ic95(int(o.sum()), len(o)) if len(o) else (np.nan,) * 1
    p_n, *_ = prop_ic95(int(na.sum()), len(na)) if len(na) else (np.nan,) * 1
    p_test = chi2_or_fisher_p(int(o.sum()), len(o), int(na.sum()), len(na))
    kpis["ramsay6"] = dict(label="% pacientes Ramsay 6", unit="%",
                            global_=p_glob, onco=p_o, noonco=p_n,
                            p=p_test, fmt="{:.1f}")

    # KPI 2 — dosis mediana eficaz (mg) en la primera visita con Ramsay 6
    g, o, na = _split(df, "dosis_primera_ram6")
    u, p_test = mw_test(o, na)
    kpis["dosis_eficaz"] = dict(label="Dosis mediana eficaz", unit="mg",
                                 global_=g.median(), onco=o.median(), noonco=na.median(),
                                 p=p_test, fmt="{:.1f}")

    # KPI 3 — tiempo mediano hasta sedacion adecuada (h)
    g, o, na = _split(df, "tiempo_hasta_ram6_h")
    u, p_test = mw_test(o, na)
    kpis["tiempo_sedacion"] = dict(label="Tiempo mediano hasta sedacion adecuada", unit="h",
                                    global_=g.median(), onco=o.median(), noonco=na.median(),
                                    p=p_test, fmt="{:.1f}")

    # KPI 4 — tasa de complicaciones del dispositivo (%)
    g, o, na = _split(df, "ea_complicacion")
    p_glob, *_ = prop_ic95(int(g.sum()), len(g))
    p_o, *_ = prop_ic95(int(o.sum()), len(o)) if len(o) else (np.nan,)
    p_n, *_ = prop_ic95(int(na.sum()), len(na)) if len(na) else (np.nan,)
    p_test = chi2_or_fisher_p(int(o.sum()), len(o), int(na.sum()), len(na))
    kpis["complicaciones"] = dict(label="Tasa de complicaciones del dispositivo", unit="%",
                                   global_=p_glob, onco=p_o, noonco=p_n,
                                   p=p_test, fmt="{:.1f}")

    # KPI 5 — tasa de rescates (%)
    g, o, na = _split(df, "uso_rescate_any")
    p_glob, *_ = prop_ic95(int(g.sum()), len(g))
    p_o, *_ = prop_ic95(int(o.sum()), len(o)) if len(o) else (np.nan,)
    p_n, *_ = prop_ic95(int(na.sum()), len(na)) if len(na) else (np.nan,)
    p_test = chi2_or_fisher_p(int(o.sum()), len(o), int(na.sum()), len(na))
    kpis["rescates"] = dict(label="Tasa de rescates", unit="%",
                             global_=p_glob, onco=p_o, noonco=p_n,
                             p=p_test, fmt="{:.1f}")

    # KPI 6 — tiempo mediano sedacion -> fallecimiento (dias)
    g, o, na = _split(df, "tiempo_fallec_dias")
    u, p_test = mw_test(o, na)
    kpis["tiempo_fallecimiento"] = dict(label="Tiempo mediano sedacion-fallecimiento", unit="dias",
                                         global_=g.median(), onco=o.median(), noonco=na.median(),
                                         p=p_test, fmt="{:.1f}")

    # KPI 7 — ratio oncologico / no oncologico
    ratio = (n_onco / n_noonco) if n_noonco else np.nan
    kpis["ratio"] = dict(label="Ratio Oncologico / No Oncologico", unit="",
                          global_=ratio, onco=n_onco, noonco=n_noonco,
                          p=np.nan, fmt="{:.2f}")

    kpis["_meta"] = dict(n_total=n_total, n_onco=n_onco, n_noonco=n_noonco)
    return kpis


def print_kpis(kpis):
    print("\n" + "=" * 70)
    print("RESUMEN DE LOS 7 KPIs (PASO 7 - DASHBOARD)")
    print("=" * 70)
    for key, k in kpis.items():
        if key == "_meta":
            continue
        g = k["global_"]
        sig = sig_label(k["p"])
        print(f"  {k['label']:48s} : global={g!s:>8} {k['unit']:5s}  "
              f"onco={k['onco']!s:>8}  noonco={k['noonco']!s:>8}  "
              f"p={fmt_p(k['p'])} ({sig})")
    print("=" * 70)


# =============================================================================
# FIGURAS PLOTLY
# =============================================================================

def _placeholder_fig(mensaje):
    fig = go.Figure()
    fig.add_annotation(
        text=f"⚠ {mensaje}",
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False,
        font=dict(color=TXT_SUB, size=14)
    )
    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT_BASE.items() if k != "margin"},
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=320
    )
    return fig


def _to_div(fig):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                        config={"displayModeBar": False, "responsive": True})


def fig_gauge_ramsay6(kpis):
    try:
        val = kpis["ramsay6"]["global_"]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0 if pd.isna(val) else val,
            number=dict(suffix=" %", font=dict(size=42, color=TXT_MAIN)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=TXT_SUB, tickfont=dict(color=TXT_SUB)),
                bar=dict(color=C_OK, thickness=0.28),
                bgcolor=BG_CARD,
                borderwidth=0,
                steps=[
                    dict(range=[0, 60], color="#FADBD8"),
                    dict(range=[60, 80], color="#FCF3CF"),
                    dict(range=[80, 100], color="#D5F5E3"),
                ],
                threshold=dict(line=dict(color=TXT_MAIN, width=3), thickness=0.8,
                                value=0 if pd.isna(val) else val),
            ),
            title=dict(text="Tasa global de consecucion de Ramsay 6",
                       font=dict(size=13, color=TXT_SUB)),
        ))
        fig.update_layout(**PLOTLY_LAYOUT_BASE, height=340)
        return fig
    except Exception as e:
        return _placeholder_fig(f"No se pudo generar el gauge Ramsay 6 ({e})")


def fig_barras_comparativas(kpis):
    """Barras horizontales agrupadas Onco vs No-Onco (escala relativa por fila,
    con el valor real anotado dentro de la barra)."""
    try:
        metricas = ["ramsay6", "dosis_eficaz", "tiempo_sedacion",
                    "complicaciones", "rescates"]
        labels = [kpis[m]["label"] for m in metricas]
        units = [kpis[m]["unit"] for m in metricas]
        onco_vals = [kpis[m]["onco"] for m in metricas]
        noonco_vals = [kpis[m]["noonco"] for m in metricas]

        onco_rel, noonco_rel, onco_txt, noonco_txt = [], [], [], []
        for ov, nv, u in zip(onco_vals, noonco_vals, units):
            mx = max(abs(ov) if pd.notna(ov) else 0, abs(nv) if pd.notna(nv) else 0, 1e-9)
            onco_rel.append(0 if pd.isna(ov) else (ov / mx) * 100)
            noonco_rel.append(0 if pd.isna(nv) else (nv / mx) * 100)
            onco_txt.append("s/d" if pd.isna(ov) else f"{ov:.1f}{u}")
            noonco_txt.append("s/d" if pd.isna(nv) else f"{nv:.1f}{u}")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=labels, x=onco_rel, orientation="h", name="Oncologico",
            marker_color=C_ONCO, text=onco_txt, textposition="inside",
            insidetextanchor="middle", textfont=dict(color="white", size=11)))
        fig.add_trace(go.Bar(
            y=labels, x=noonco_rel, orientation="h", name="No Oncologico",
            marker_color=C_NOONCO, text=noonco_txt, textposition="inside",
            insidetextanchor="middle", textfont=dict(color="white", size=11)))
        fig.update_layout(
            **PLOTLY_LAYOUT_BASE, barmode="group", height=420,
            title=dict(text="Comparativa Oncologico vs No Oncologico · KPIs principales",
                       font=dict(size=14, color=TXT_MAIN)),
            xaxis=dict(title="Valor relativo por KPI (%)", gridcolor=GRID, range=[0, 115]),
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", y=-0.12, font=dict(color=TXT_MAIN)),
        )
        return fig
    except Exception as e:
        return _placeholder_fig(f"No se pudo generar la comparativa de KPIs ({e})")


def fig_violin_dosis(df):
    try:
        sub = df.dropna(subset=["dosis_acum_mid", "tipo_label"])
        if sub.empty:
            return _placeholder_fig("Sin datos de dosis acumulada de midazolam")
        fig = go.Figure()
        for tipo, color in ((1, C_ONCO), (2, C_NOONCO)):
            d = sub.loc[sub["Tipo de paciente"] == tipo, "dosis_acum_mid"]
            if d.empty:
                continue
            fig.add_trace(go.Violin(
                y=d, x=[LABEL_TIPO[tipo]] * len(d), name=LABEL_TIPO[tipo],
                line_color=color, fillcolor=color, opacity=0.55,
                points="all", jitter=0.35, pointpos=0, marker=dict(size=5, color=color),
                box_visible=True, meanline_visible=True))
            fig.add_annotation(x=LABEL_TIPO[tipo], y=d.median(),
                                text=f"Mediana: {d.median():.1f} mg",
                                showarrow=True, arrowhead=2, ax=40, ay=-30,
                                font=dict(size=10, color=TXT_MAIN),
                                arrowcolor=TXT_SUB)
        fig.update_layout(
            **PLOTLY_LAYOUT_BASE, height=420, showlegend=False,
            title=dict(text="Distribucion de dosis acumulada de midazolam por perfil",
                       font=dict(size=14, color=TXT_MAIN)),
            yaxis=dict(title="Dosis acumulada (mg)", gridcolor=GRID, range=[0, 200]),
            xaxis=dict(gridcolor=GRID),
        )
        n_fuera = int((sub["dosis_acum_mid"] > 200).sum())
        if n_fuera > 0:
            fig.add_annotation(
                text=f"Nota: {n_fuera} caso(s) con dosis >200 mg fuera del rango visible",
                xref="paper", yref="paper", x=0.5, y=1.08, showarrow=False,
                font=dict(size=10, color=TXT_SUB))
        return fig
    except Exception as e:
        return _placeholder_fig(f"No se pudo generar el violin de dosis ({e})")


def fig_terciles_kw(df):
    try:
        sub = df.dropna(subset=["dosis_acum_mid", "tiempo_fallec_dias"]).copy()
        if len(sub) < 6:
            return _placeholder_fig("Datos insuficientes para el analisis por tercil")
        sub["tiempo_fallec_horas"] = sub["tiempo_fallec_dias"] * 24
        try:
            sub["tercil"] = pd.qcut(sub["dosis_acum_mid"], 3, duplicates="drop")
        except Exception:
            sub["tercil"] = pd.cut(sub["dosis_acum_mid"], 3)
        grupos = [g["tiempo_fallec_horas"].dropna().values
                  for _, g in sub.groupby("tercil")]
        grupos = [g for g in grupos if len(g) >= 1]
        if len(grupos) >= 2:
            stat, p_kw = kruskal(*grupos)
        else:
            stat, p_kw = np.nan, np.nan

        resumen = sub.groupby("tercil", observed=True)["tiempo_fallec_horas"].median()
        cats = [f"T{i+1}\n({iv.left:.0f}-{iv.right:.0f} mg)"
                for i, iv in enumerate(resumen.index)]

        fig = go.Figure(go.Bar(
            x=cats, y=resumen.values, marker_color=C_WARN,
            text=[f"{v:.0f} h" for v in resumen.values], textposition="outside",
            textfont=dict(color=TXT_MAIN)))
        sig = sig_label(p_kw)
        subt = (f"Kruskal-Wallis: p={fmt_p(p_kw)} ({sig})" if not pd.isna(p_kw)
                else "Kruskal-Wallis: no calculable")
        fig.update_layout(
            **PLOTLY_LAYOUT_BASE, height=420,
            title=dict(text=f"Tiempo hasta fallecimiento por tercil de dosis acumulada<br>"
                            f"<span style='font-size:11px;color={TXT_SUB}'>{subt}</span>",
                       font=dict(size=14, color=TXT_MAIN)),
            yaxis=dict(title="Tiempo mediano hasta fallecimiento (horas)", gridcolor=GRID),
            xaxis=dict(gridcolor=GRID),
        )
        return fig
    except Exception as e:
        return _placeholder_fig(f"No se pudo generar el analisis por tercil ({e})")


def fig_donut_ratio(kpis):
    try:
        n_o = kpis["_meta"]["n_onco"]
        n_n = kpis["_meta"]["n_noonco"]
        fig = go.Figure(go.Pie(
            values=[n_o, n_n], labels=["Onco", "No Onco"], hole=0.65,
            marker=dict(colors=[C_ONCO, C_NOONCO]), textinfo="none", sort=False))
        ratio = kpis["ratio"]["global_"]
        ratio_txt = "s/d" if pd.isna(ratio) else f"{ratio:.2f}"
        fig.add_annotation(text=f"<b>{ratio_txt}</b>", x=0.5, y=0.5, showarrow=False,
                            font=dict(size=20, color=TXT_MAIN))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(family=FONT_FAM, color=TXT_MAIN),
                           margin=dict(l=0, r=0, t=0, b=0), height=90, width=90,
                           showlegend=False)
        return fig
    except Exception as e:
        return _placeholder_fig(f"Ratio no disponible ({e})")


def fig_tabla_resumen(kpis):
    try:
        rows = ["ramsay6", "dosis_eficaz", "tiempo_sedacion",
                "complicaciones", "rescates", "tiempo_fallecimiento", "ratio"]
        col_kpi, col_glob, col_onco, col_noonco, col_sig, col_sigcolor = [], [], [], [], [], []
        for r in rows:
            k = kpis[r]
            u = k["unit"]
            fmt = k["fmt"]

            def _f(v):
                return "s/d" if pd.isna(v) else f"{fmt.format(v)}{(' ' + u) if u else ''}"

            col_kpi.append(k["label"])
            col_glob.append(_f(k["global_"]))
            if r == "ratio":
                col_onco.append(f"n={k['onco']}")
                col_noonco.append(f"n={k['noonco']}")
                col_sig.append("n/a")
                col_sigcolor.append(TXT_SUB)
            else:
                col_onco.append(_f(k["onco"]))
                col_noonco.append(_f(k["noonco"]))
                s = sig_label(k["p"])
                col_sig.append(f"{s} (p={fmt_p(k['p'])})")
                col_sigcolor.append(sig_badge_color(s))

        n_rows = len(rows)
        row_colors = [BG_CARD if i % 2 == 0 else "#FBDCEA" for i in range(n_rows)]

        fig = go.Figure(go.Table(
            columnwidth=[34, 16, 16, 16, 22],
            header=dict(
                values=["<b>KPI</b>", "<b>Global</b>", "<b>Oncologico</b>",
                        "<b>No Oncologico</b>", "<b>Significacion</b>"],
                fill_color=TXT_MAIN, font=dict(color="#FFFFFF", size=12), height=34,
                align="left"),
            cells=dict(
                values=[col_kpi, col_glob, col_onco, col_noonco, col_sig],
                fill_color=[row_colors] * 5,
                font=dict(color=[[TXT_MAIN] * n_rows, [TXT_MAIN] * n_rows,
                                  [C_ONCO] * n_rows, [C_NOONCO] * n_rows,
                                  col_sigcolor], size=11),
                align="left", height=30),
        ))
        base = {k: v for k, v in PLOTLY_LAYOUT_BASE.items() if k != "margin"}
        fig.update_layout(**base, height=320,
                           margin=dict(l=10, r=10, t=10, b=10))
        return fig
    except Exception as e:
        return _placeholder_fig(f"No se pudo generar la tabla resumen ({e})")


# =============================================================================
# TARJETAS KPI (HTML/CSS)
# =============================================================================

def _kpi_color_complicaciones(pct):
    if pd.isna(pct):
        return TXT_SUB
    if pct > 10:
        return C_ONCO
    if pct >= 5:
        return C_WARN
    return C_OK


def _card(icon, valor, unidad, etiqueta, color, sub_onco="", sub_noonco="", extra_html=""):
    sub = ""
    if sub_onco or sub_noonco:
        sub = (f'<div class="kpi-sub">'
               f'<span style="color:{C_ONCO}">{sub_onco}</span> &nbsp;·&nbsp; '
               f'<span style="color:{C_NOONCO}">{sub_noonco}</span></div>')
    return f"""
    <div class="kpi-card" style="border-top-color:{color}">
      <div class="kpi-icon" style="color:{color}">{icon}</div>
      <div class="kpi-valor" style="color:{color}">{valor}<span class="kpi-unidad">{unidad}</span></div>
      <div class="kpi-label">{etiqueta}</div>
      {sub}
      {extra_html}
    </div>"""


def build_kpi_cards(kpis):
    k = kpis
    fmt1 = lambda v: ("s/d" if pd.isna(v) else f"{v:.1f}")

    c1 = _card("🛏", fmt1(k["ramsay6"]["global_"]), "%", "Pacientes que alcanzan Ramsay 6",
                C_OK, f"Onco {fmt1(k['ramsay6']['onco'])}%", f"No-Onco {fmt1(k['ramsay6']['noonco'])}%")
    c2 = _card("💧", fmt1(k["dosis_eficaz"]["global_"]), " mg", "Dosis mediana eficaz",
                C_WARN, f"Onco {fmt1(k['dosis_eficaz']['onco'])}mg", f"No-Onco {fmt1(k['dosis_eficaz']['noonco'])}mg")
    c3 = _card("⏱", fmt1(k["tiempo_sedacion"]["global_"]), " h",
                "Tiempo mediano hasta sedacion adecuada",
                C_NOONCO, f"Onco {fmt1(k['tiempo_sedacion']['onco'])}h", f"No-Onco {fmt1(k['tiempo_sedacion']['noonco'])}h")
    color4 = _kpi_color_complicaciones(k["complicaciones"]["global_"])
    c4 = _card("⚠", fmt1(k["complicaciones"]["global_"]), "%", "Tasa de complicaciones del dispositivo",
                color4, f"Onco {fmt1(k['complicaciones']['onco'])}%", f"No-Onco {fmt1(k['complicaciones']['noonco'])}%")
    c5 = _card("💉", fmt1(k["rescates"]["global_"]), "%", "Tasa de rescates",
                C_WARN, f"Onco {fmt1(k['rescates']['onco'])}%", f"No-Onco {fmt1(k['rescates']['noonco'])}%")
    c6 = _card("🕊", fmt1(k["tiempo_fallecimiento"]["global_"]), " d",
                "Tiempo mediano sedacion-fallecimiento",
                C_NOONCO, f"Onco {fmt1(k['tiempo_fallecimiento']['onco'])}d", f"No-Onco {fmt1(k['tiempo_fallecimiento']['noonco'])}d")

    donut_div = _to_div(fig_donut_ratio(k))
    ratio = k["ratio"]["global_"]
    ratio_txt = "s/d" if pd.isna(ratio) else f"{ratio:.2f} : 1"
    c7 = f"""
    <div class="kpi-card kpi-card-donut" style="border-top-color:{TXT_SUB}">
      <div class="kpi-donut-wrap">{donut_div}</div>
      <div class="kpi-valor" style="color:{TXT_MAIN};font-size:1.05rem">{ratio_txt}</div>
      <div class="kpi-label">Ratio Oncologico / No Oncologico</div>
      <div class="kpi-sub"><span style="color:{C_ONCO}">n={k['_meta']['n_onco']}</span> &nbsp;·&nbsp;
      <span style="color:{C_NOONCO}">n={k['_meta']['n_noonco']}</span></div>
    </div>"""

    return "".join([c1, c2, c3, c4, c5, c6, c7])


# =============================================================================
# ENSAMBLADO HTML
# =============================================================================

CSS = f"""
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 0; background: {BG_PAGE}; color: {TXT_MAIN};
  font-family: {FONT_FAM};
}}
.wrap {{ max-width: 1500px; margin: 0 auto; padding: 28px 24px 50px; }}
.header {{
  display: flex; justify-content: space-between; align-items: flex-start;
  flex-wrap: wrap; gap: 14px; padding-bottom: 22px; margin-bottom: 22px;
  border-bottom: 1px solid {GRID};
}}
.header h1 {{ margin: 0 0 6px; font-size: 1.7rem; font-weight: 700; color: {TXT_MAIN}; }}
.header p {{ margin: 0; color: {TXT_SUB}; font-size: 0.95rem; }}
.badge {{
  background: {BG_CARD}; border: 1px solid {GRID}; border-radius: 10px;
  padding: 10px 18px; text-align: right; min-width: 180px;
}}
.badge .n {{ font-size: 1.5rem; font-weight: 700; color: {C_OK}; }}
.badge .d {{ font-size: 0.78rem; color: {TXT_SUB}; }}
.kpi-row {{
  display: grid; grid-template-columns: repeat(7, 1fr); gap: 14px; margin-bottom: 28px;
}}
@media (max-width: 1300px) {{ .kpi-row {{ grid-template-columns: repeat(4, 1fr); }} }}
@media (max-width: 760px) {{ .kpi-row {{ grid-template-columns: repeat(2, 1fr); }} }}
.kpi-card {{
  background: {BG_CARD}; border-top: 4px solid; border-radius: 10px;
  padding: 18px 16px; min-height: 168px; display: flex; flex-direction: column;
  justify-content: center; box-shadow: 0 2px 10px rgba(16,24,40,0.08);
}}
.kpi-card-donut {{ align-items: center; text-align: center; }}
.kpi-donut-wrap {{ display: flex; justify-content: center; }}
.kpi-icon {{ font-size: 1.6rem; margin-bottom: 5px; }}
.kpi-valor {{ font-size: 2.3rem; font-weight: 700; line-height: 1.1; }}
.kpi-unidad {{ font-size: 1.15rem; font-weight: 500; margin-left: 2px; }}
.kpi-label {{ font-size: 0.92rem; color: {TXT_SUB}; margin-top: 7px; line-height: 1.3; }}
.kpi-sub {{ font-size: 0.86rem; margin-top: 9px; font-weight: 600; }}
.row {{ display: grid; gap: 18px; margin-bottom: 22px; }}
.row-60-40 {{ grid-template-columns: 60% 1fr; }}
.row-50-50 {{ grid-template-columns: 1fr 1fr; }}
@media (max-width: 1000px) {{ .row-60-40, .row-50-50 {{ grid-template-columns: 1fr; }} }}
.panel {{
  background: {BG_CARD}; border-radius: 12px; padding: 14px;
  box-shadow: 0 2px 10px rgba(16,24,40,0.08);
}}
.footer {{
  margin-top: 30px; padding-top: 18px; border-top: 1px solid {GRID};
  text-align: center; color: {TXT_SUB}; font-size: 0.82rem; line-height: 1.6;
}}
"""


def main():
    df, e2 = load_data()
    kpis = compute_kpis(df)

    cards_html = build_kpi_cards(kpis)
    div_bar = _to_div(fig_barras_comparativas(kpis))
    div_gauge = _to_div(fig_gauge_ramsay6(kpis))
    div_violin = _to_div(fig_violin_dosis(df))
    div_terciles = _to_div(fig_terciles_kw(df))
    div_tabla = _to_div(fig_tabla_resumen(kpis))

    n_total = kpis["_meta"]["n_total"]
    fecha_gen = datetime.now().strftime("%d/%m/%Y %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Cuadro de Mandos · Sedacion Paliativa con Infusores</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

  <div class="header">
    <div>
      <h1>Cuadro de Mandos · Sedacion Paliativa con Infusores</h1>
      <p>Hospitalizacion a Domicilio — Hospital Universitario La Fe · Valencia</p>
    </div>
    <div class="badge">
      <div class="n">n = {n_total}</div>
      <div class="d">Generado: {fecha_gen}</div>
    </div>
  </div>

  <div class="kpi-row">
    {cards_html}
  </div>

  <div class="row row-60-40">
    <div class="panel">{div_bar}</div>
    <div class="panel">{div_gauge}</div>
  </div>

  <div class="row row-50-50">
    <div class="panel">{div_violin}</div>
    <div class="panel">{div_terciles}</div>
  </div>

  <div class="row" style="grid-template-columns: 1fr;">
    <div class="panel">{div_tabla}</div>
  </div>

  <div class="footer">
    Estudio INFUSED-UHD · TFG Ingenieria Biomedica · Universitat Politecnica de Valencia<br/>
    Analisis restringido a pacientes con consentimiento informado (n={n_total}).
    Datos anonimizados conforme al RGPD.
  </div>

</div>
</body>
</html>
"""

    fname = out("P7_Dashboard_Sedacion_Paliativa.html")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(html)

    print_kpis(kpis)
    print(f"\n  -> Dashboard guardado en: {fname}")
    return kpis


if __name__ == "__main__":
    main()
