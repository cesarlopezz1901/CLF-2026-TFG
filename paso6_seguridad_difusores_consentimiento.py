"""
=============================================================================
PASO 6 - SEGURIDAD DE LOS DIFUSORES (EVENTOS ADVERSOS DEL INFUSOR)
TFG: Manejo del Paciente Paliativo en Sedacion Continua - Hospital La Fe
=============================================================================
Solo pacientes con Consentimiento Informado (CI=1).

Reaprovecha integramente las funciones de paso6_seguridad_difusores.py,
cambiando el directorio de salida y aplicando el filtro CI=1 al construir
df_clean. Mantiene el mismo patron usado en los Pasos 3, 4 y 5.

Salida: resultados_paso6_CI/
=============================================================================
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Respeta OUTPUT_DIR si ya viene del entorno; si no, usa el valor por defecto
if not os.environ.get("OUTPUT_DIR"):
    os.environ["OUTPUT_DIR"] = os.path.join(_SCRIPT_DIR, "resultados_ANONIMIZADO_CI")
os.makedirs(os.environ["OUTPUT_DIR"], exist_ok=True)

# Anade ruta del modulo principal al PYTHONPATH
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Importa todo el modulo principal y sobreescribe la etiqueta de cohorte
import paso6_seguridad_difusores as p6  # noqa: E402

p6.COHORTE = "Solo CI=1"


def main():
    """Ejecuta el Paso 6 sobre la cohorte filtrada por CI=1."""
    print(f"\nCargando datos (cohorte: {p6.COHORTE} - filtro CI=1)...")
    resultados = p6.seguridad_difusores(filtro_ci=True)
    print("\n" + "=" * 60)
    print("PASO 6 (CI=1) COMPLETADO")
    print(f"  Salidas en: {p6.OUTPUT_DIR}")
    print("=" * 60)
    return resultados


if __name__ == "__main__":
    main()
