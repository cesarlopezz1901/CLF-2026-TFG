# CLF-2026-TFG · INFUSED-UHD

Pipeline de análisis estadístico del Trabajo de Fin de Grado **"Análisis de la sedación paliativa con infusores en Hospitalización a Domicilio del Hospital La Fe: efectividad, seguridad y comparativa por perfil diagnóstico"**.

Desarrollado en el marco de la colaboración entre la **Universitat Politècnica de València** y la **Unidad de Hospitalización a Domicilio del Hospital Universitario y Politécnico La Fe**, con aprobación del comité de ética correspondiente.

---

## Estructura del pipeline

| Script | Descripción |
|--------|-------------|
| `anonimizar_final.py` | Depuración de identificadores directos y seudonimización mediante cifrado Fernet. Genera los códigos de estudio KT001–KT050 |
| `paso3_descriptivo2_consentimiento.py` | Perfil sociodemográfico y clínico de la cohorte. Comparativa oncológico vs. no oncológico |
| `paso4_dosis_eficacia_consentimiento.py` | Análisis dosis-respuesta, tiempos clínicos de sedación y gestión de rescates |
| `paso5_estadistica_avanzada_consentimiento.py` | Análisis de componentes principales, regresión logística y lineal, contraste de hipótesis y análisis de subgrupos |
| `paso6_seguridad_difusores_consentimiento.py` | Identificación y clasificación de eventos adversos asociados al infusor. Análisis de supervivencia |
| `paso7_dashboard.py` | Construcción de KPIs clínicos del proceso de sedación y generación del dashboard interactivo |

---

## Carpeta RESULTS

Contiene las figuras, tablas y ficheros CSV generados por el pipeline e integrados en la memoria del TFG. Los ficheros se nombran siguiendo el esquema `P{paso}_{tipo}_{descripcion}`.

---

## Datos

Los ficheros de datos originales y anonimizados **no se incluyen** en este repositorio por motivos de protección de datos (RGPD, Reglamento UE 2016/679, y LOPGDD, Ley Orgánica 3/2018). El código se publica exclusivamente con fines de transparencia metodológica y reproducibilidad del análisis.

---

## Dependencias principales

```text
pandas · numpy · scipy · statsmodels · scikit-learn
matplotlib · seaborn · plotly · lifelines · cryptography
```

---

## Autor

**César López Ferrández** · Grado en Ingeniería Biomédica · UPV · Curso 2025-2026  
Supervisión clínica: Dra. María Elena Castro Vilela · HaD, Hospital Universitario y Politécnico La Fe
