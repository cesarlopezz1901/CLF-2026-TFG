"""
Anonimización de EXCEL_1_FINAL y EXCEL_2_FINAL con Fernet (cifrado reversible).

Salidas para análisis:
  EXCEL_1_ANONIMIZADO_FINAL.xlsx
  EXCEL_2_ANONIMIZADO_FINAL.xlsx

Salidas CONFIDENCIALES (no subir a repositorios):
  archivo_tokens_final.xlsx  — correspondencia NHC ↔ token cifrado ↔ código KT
  clave.key                  — clave Fernet (ya existente, no se regenera)
"""

import os
import pandas as pd
from cryptography.fernet import Fernet

EXCEL1      = "EXCEL_1_FINAL.xlsx"
EXCEL2      = "EXCEL_2_FINAL.xlsx"
CLAVE_FILE  = "clave.key"
TOKENS_FILE = "archivo_tokens_final.xlsx"
OUT1        = "EXCEL_1_ANONIMIZADO_FINAL.xlsx"
OUT2        = "EXCEL_2_ANONIMIZADO_FINAL.xlsx"

# ── 1. Cargar excels ──────────────────────────────────────────────────────────
print("Cargando Excel 1 FINAL...")
df1 = pd.read_excel(EXCEL1)
print(f"  {len(df1)} filas | columnas: {list(df1.columns[:5])}...")

print("Cargando Excel 2 FINAL...")
df2 = pd.read_excel(EXCEL2)
print(f"  {len(df2)} filas | columnas: {list(df2.columns[:5])}...")

# ── 2. Cargar clave Fernet existente ─────────────────────────────────────────
if not os.path.exists(CLAVE_FILE):
    raise FileNotFoundError(
        f"No se encontró '{CLAVE_FILE}'. "
        "Coloca la clave en el mismo directorio que este script."
    )

print(f"\nCargando clave desde '{CLAVE_FILE}'...")
with open(CLAVE_FILE, "rb") as f:
    clave = f.read()
fernet = Fernet(clave)

# ── 3. Extraer NHC únicos de ambos ficheros combinados ───────────────────────
print("\nExtrayendo NHC únicos de los dos ficheros...")
nhc_unicos_set = (
    set(df1["Nhc"].dropna().astype(str).unique()) |
    set(df2["Nhc"].dropna().astype(str).unique())
)
nhc_unicos = sorted(nhc_unicos_set)
print(f"  {len(nhc_unicos)} pacientes únicos encontrados.")

# ── 4. Cifrar cada NHC con Fernet ────────────────────────────────────────────
print("Cifrando NHC...")
tokens = [fernet.encrypt(nhc.encode()).decode() for nhc in nhc_unicos]

# ── 5. Asignar código secuencial KT001, KT002... con pd.factorize ────────────
print("Asignando codigos KT...")
import numpy as np
indices, _ = pd.factorize(np.array(nhc_unicos), sort=True)
codigos = [f"KT{i + 1:03d}" for i in indices]

mapeo_nhc_codigo = dict(zip(nhc_unicos, codigos))
mapeo_nhc_token  = dict(zip(nhc_unicos, tokens))

print("  Primeros 5 ejemplos:")
for nhc in nhc_unicos[:5]:
    print(f"    NHC {nhc} -> {mapeo_nhc_codigo[nhc]}  (token: {mapeo_nhc_token[nhc][:20]}...)")

# ── 6. Guardar archivo de tokens CONFIDENCIAL ────────────────────────────────
df_tokens = pd.DataFrame({
    "NHC":           nhc_unicos,
    "token_cifrado": tokens,
    "codigo":        codigos,
})
df_tokens.to_excel(TOKENS_FILE, index=False)
print(f"\nTokens guardados en '{TOKENS_FILE}' (CONFIDENCIAL).")

# ── 7. Anonimizar Excel 1 ────────────────────────────────────────────────────
print("\nAnonimizando Excel 1...")
df1_anon = df1.copy()
df1_anon["codigo"] = df1_anon["Nhc"].astype(str).map(mapeo_nhc_codigo)

sin_codigo = df1_anon["codigo"].isna().sum()
if sin_codigo:
    print(f"  AVISO: {sin_codigo} filas con NHC sin código (pueden ser NaN en origen).")

# Eliminar columnas identificadoras
cols_a_eliminar_e1 = [c for c in ["Nhc", "fecanac"] if c in df1_anon.columns]
df1_anon = df1_anon.drop(columns=cols_a_eliminar_e1)

# Columna 'codigo' en primera posición
df1_anon = df1_anon[["codigo"] + [c for c in df1_anon.columns if c != "codigo"]]
print(f"  Columnas: {list(df1_anon.columns[:6])}...")

# ── 8. Anonimizar Excel 2 ────────────────────────────────────────────────────
print("Anonimizando Excel 2...")
df2_anon = df2.copy()
df2_anon["codigo"] = df2_anon["Nhc"].astype(str).map(mapeo_nhc_codigo)

sin_codigo2 = df2_anon["codigo"].isna().sum()
if sin_codigo2:
    print(f"  AVISO: {sin_codigo2} filas con NHC sin código.")

cols_a_eliminar_e2 = [c for c in ["Nhc"] if c in df2_anon.columns]
df2_anon = df2_anon.drop(columns=cols_a_eliminar_e2)

df2_anon = df2_anon[["codigo"] + [c for c in df2_anon.columns if c != "codigo"]]
print(f"  Columnas: {list(df2_anon.columns[:6])}...")

# ── 9. Limpiar fechas y horas para no exponer timestamps exactos ─────────────
print("\nLimpiando formato de fechas y horas...")

FECHAS_E1 = ["Fecha", "Fecha inicio sedación", "Fecha fallecimiento",
             "Fecha inicio sed"]
HORAS_E1  = ["Hora inicio sedación", "Hora éxitus", "Hora inicio sed"]

FECHAS_E2 = ["Fecha visita", "Fecha cambio infusor", "Fecha reemplazo de infusor"]
HORAS_E2  = ["Hora visita", "Hora enf", "Hora reemplazo infusor"]

def limpiar_fechas(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = (pd.to_datetime(df[col], errors="coerce")
                       .dt.strftime("%d/%m/%Y").fillna(""))
    return df

def limpiar_horas(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str[:5].replace("nan", "")
    return df

df1_anon = limpiar_fechas(df1_anon, FECHAS_E1)
df1_anon = limpiar_horas(df1_anon, HORAS_E1)
df2_anon = limpiar_fechas(df2_anon, FECHAS_E2)
df2_anon = limpiar_horas(df2_anon, HORAS_E2)

# ── 10. Guardar ficheros anonimizados ────────────────────────────────────────
print(f"\nGuardando '{OUT1}'...")
df1_anon.to_excel(OUT1, index=False)

print(f"Guardando '{OUT2}'...")
df2_anon.to_excel(OUT2, index=False)

# ── Resumen ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANONIMIZACIÓN COMPLETADA")
print("=" * 60)
print(f"  Pacientes únicos procesados : {len(nhc_unicos)}")
print(f"  Filas en Excel 1            : {len(df1_anon)}")
print(f"  Filas en Excel 2            : {len(df2_anon)}")
print()
print("Archivos para análisis (sin identificadores):")
print(f"  [OK] {OUT1}")
print(f"  [OK] {OUT2}")
print()
print("Archivos CONFIDENCIALES (solo custodia del investigador):")
print(f"  [!!] {TOKENS_FILE}  -- NHC <-> token cifrado <-> código KT")
print(f"  [!!] {CLAVE_FILE}         -- clave Fernet (NUNCA compartir ni subir)")
