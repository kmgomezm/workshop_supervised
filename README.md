# 🤖 ML Supervisado · EAFIT 2026

**Taller 02 — Aprendizaje Supervisado: Regresión & Clasificación**  
Maestría en Ciencia de Datos · Docente: Jorge I. Padilla-Buriticá

---

## 🗂️ Estructura del Repositorio

```
ml_supervisado/
├── data/
│   ├── raw/
│   │   ├── insurance.csv                          ← Dataset regresión
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  ← Dataset clasificación
│   └── processed/                                 ← Generado por notebooks
├── notebooks/
│   ├── 🏥 REGRESIÓN — Medical Insurance
│   │   ├── 01_EDA.ipynb
│   │   ├── 02_Preprocessing.ipynb
│   │   ├── 03_Feature_Engineering.ipynb
│   │   ├── 04_Model_Training.ipynb
│   │   └── 05_Validation.ipynb
│   └── 📡 CLASIFICACIÓN — Telco Churn
│       ├── 06_Churn_EDA.ipynb
│       ├── 07_Churn_Preprocessing.ipynb
│       ├── 08_Churn_Feature_Engineering.ipynb
│       ├── 09_Churn_Model_Training.ipynb
│       └── 10_Churn_Validation.ipynb
├── models/                                        ← .pkl generados por notebooks
├── app/
│   └── app.py                                     ← Streamlit unificado
├── requirements.txt
└── README.md
```

---

## 🏥 Parte 1: Regresión — Medical Insurance

**Dataset:** Medical Cost Personal Dataset (1,338 registros, 7 columnas)  
**Target:** `charges` (costo médico anual en USD)

| Notebook | Contenido |
|----------|-----------|
| `01_EDA` | Distribuciones, correlaciones, análisis por smoker/región |
| `02_Preprocessing` | Limpieza, OHE, LabelEncoding, StandardScaler, split 80/20 |
| `03_Feature_Engineering` | Variance Threshold, MI, SelectKBest, LASSO, RF Importance |
| `04_Model_Training` | Ridge, KNN, Random Forest + GridSearch/RandomSearch |
| `05_Validation` | CV KFold-5, curvas de aprendizaje, reporte final |

**Modelos:** Ridge Regression · KNN Regressor · Random Forest  
**Target transform:** `log1p(charges)` → `expm1()` para métricas en $  
**Métricas:** RMSE, MAE, R²

---

## 📡 Parte 2: Clasificación — Telco Customer Churn

**Dataset:** IBM Telco Customer Churn (7,043 registros, 21 columnas)  
**Target:** `Churn` (Yes=1 / No=0) — Clasificación binaria desbalanceada (~27% Churn)

| Notebook | Contenido |
|----------|-----------|
| `06_Churn_EDA` | Análisis de desbalance, Chi², tasa de churn por variable |
| `07_Churn_Preprocessing` | Codificación binaria/ternaria/OHE, split estratificado |
| `08_Churn_Feature_Engineering` | MI Classif., Chi², RF Importance, Logistic L1 |
| `09_Churn_Model_Training` | LogReg, KNN, Random Forest + GridSearch |
| `10_Churn_Validation` | StratifiedKFold-5, ROC, PR-AUC, análisis de umbral |

**Modelos:** Logistic Regression (L2) · KNN Classifier · Random Forest  
**Desbalance:** `class_weight='balanced'` en todos los modelos  
**Métricas:** F1-Score, AUC-ROC, Precision, Recall, PR-AUC

---

## 🚀 Dashboard Interactivo (Streamlit)

App unificada con selector de tarea (Regresión / Clasificación):

| Vista | Descripción |
|-------|-------------|
| 🎯 Predicción Individual | Formulario interactivo + factores de riesgo |
| 📂 Predicción por Lote | Upload CSV + descarga resultados |
| 📊 Dashboard Modelos | Métricas comparativas + CV con error bars |
| 🔍 Feature Importance | Gráfico RF + barra de importancia (✅ checklist) |
| 📈 Análisis Dataset | Distribuciones, correlación, segmentación |

### Ejecutar localmente:
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

### Ejecutar en Streamlit Cloud:
1. Haz fork de este repositorio
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repo → Main file: `app/app.py`

---

## 📋 Orden de Ejecución de Notebooks

```
1. Coloca los datasets en data/raw/
2. Ejecuta 01 → 02 → 03 → 04 → 05   (Regresión)
3. Ejecuta 06 → 07 → 08 → 09 → 10   (Clasificación)
4. streamlit run app/app.py
```

---

## ✅ Checklist del Taller

- [x] Repositorio con carpetas `/data`, `/notebooks`, `/app`
- [x] 10 notebooks comentados con conclusiones por sección
- [x] `requirements.txt` completo
- [x] 3+ modelos por tarea con GridSearch/RandomSearch
- [x] EDA: correlaciones, valores faltantes, distribuciones
- [x] Preprocesamiento: OHE + LabelEncoding + StandardScaler
- [x] Ingeniería de características: 4 métodos de selección
- [x] Cross-Validation + métricas en Test set
- [x] Dashboard con Feature Importance (gráfico RF)
- [x] Predicción individual + predicción por lote CSV
- [x] Desbalance de clases: `class_weight='balanced'`

---

> *"In God we trust, all others must bring data."* — W. Edwards Deming
