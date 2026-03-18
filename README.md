# ML Supervisado · EAFIT 2026

**Taller 02 — Aprendizaje Supervisado: Regresión & Clasificación**  
Maestría en Ciencia de Datos · Docente: Jorge I. Padilla-Buriticá · Período 2026-1

**Integrantes:**
- Ana Patricia Montes Pimienta
- Karen Melissa Gómez Montoya
- Juan Esteban Estrada Herrera

---

## Estructura del Repositorio

```
ml_supervisado/
├── data/
│   ├── raw/
│   │   ├── insurance.csv                          ← Dataset regresión (1,338 registros)
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  ← Dataset clasificación (7,043 registros)
│   └── processed/                                 ← Generado automáticamente por los notebooks
├── notebooks/
│   ├──  REGRESIÓN — Medical Insurance
│   │   ├── 01_EDA.ipynb
│   │   ├── 02_Preprocessing.ipynb
│   │   ├── 03_Feature_Engineering.ipynb
│   │   ├── 04_Model_Training.ipynb
│   │   └── 05_Validation.ipynb
│   └──  CLASIFICACIÓN — Telco Churn
│       ├── 06_Churn_EDA.ipynb
│       ├── 07_Churn_Preprocessing.ipynb
│       ├── 08_Churn_Feature_Engineering.ipynb
│       ├── 09_Churn_Model_Training.ipynb
│       └── 10_Churn_Validation.ipynb
├── models/                                        ← Archivos .pkl generados por los notebooks
│   ├── linear_regression.pkl
│   ├── knn_regressor.pkl
│   ├── random_forest.pkl
│   ├── churn_logistic.pkl
│   ├── churn_knn.pkl
│   └── churn_rf.pkl
├── app/
│   └── app.py                                     ← Dashboard Streamlit unificado
├── requirements.txt
└── README.md
```

---

## Parte 1: Regresión — Medical Insurance

**Dataset:** Medical Cost Personal Dataset  
**Registros:** 1,338 · **Variables:** 7 · **Target:** `charges` (costo médico anual en USD)

### Variables del dataset
| Variable | Tipo | Descripción |
|----------|------|-------------|
| `age` | Numérica | Edad del beneficiario |
| `sex` | Categórica | Género (female / male) |
| `bmi` | Numérica | Índice de masa corporal |
| `children` | Numérica | Número de hijos cubiertos |
| `smoker` | Binaria | Fumador (yes / no) |
| `region` | Categórica | Región residencial en EE.UU. |
| `charges` | **TARGET** | Costos médicos anuales (USD) |

### Flujo de Notebooks

| Notebook | Contenido principal |
|----------|---------------------|
| `01_EDA` | Distribuciones, skewness, Q-Q plots, correlaciones Pearson/Spearman, ANOVA, impacto de `smoker` en charges. Sin valores faltantes. |
| `02_Preprocessing` | Eliminación de duplicados, validación de rangos, OHE para `sex`, `region`, `bmi_category`, `age_group`; Label Encoding para `smoker`; features de interacción `bmi_smoker` y `age_smoker`; transformación `log1p(charges)`; split 80/20. |
| `03_Feature_Engineering` | Variance Threshold, Mutual Information (regresión), Prueba F (SelectKBest), RF Importance, LASSO (L1). Ranking combinado para selección final. |
| `04_Model_Training` | Ridge (GridSearchCV sobre `alpha`), KNN Regressor (curva K vs RMSE + GridSearchCV), Random Forest (RandomizedSearchCV + fine-tune GridSearch). Métricas en escala original con `expm1()`. |
| `05_Validation` | KFold-5 y KFold-10 CV, boxplots de distribución de métricas, curvas de aprendizaje, análisis de residuales, tabla comparativa final. |

### Modelos entrenados
| Modelo | Hiperparámetro principal | Búsqueda |
|--------|--------------------------|----------|
| Ridge Regression | `alpha` ∈ {0.001 … 1000} | GridSearchCV |
| KNN Regressor | `k`, `weights`, `metric` | GridSearchCV |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_*`, `max_features` | RandomizedSearch + GridSearch |

**Transform:** `log1p(charges)` para entrenamiento → `expm1()` para métricas en USD  
**Métricas:** RMSE · MAE · MSE · R²  
**Validación:** KFold-5 y KFold-10 con intervalos de confianza

---

## Parte 2: Clasificación — Telco Customer Churn

**Dataset:** IBM Telco Customer Churn  
**Registros:** 7,043 · **Variables:** 21 · **Target:** `Churn` (Yes=1 / No=0)  
**Desbalance:** ~26.5% Churn=Yes / ~73.5% Churn=No

### Flujo de Notebooks

| Notebook | Contenido principal |
|----------|---------------------|
| `06_Churn_EDA` | Análisis del desbalance de clases (ratio 2.7:1), distribuciones de `tenure`, `MonthlyCharges`, `TotalCharges` por clase, tasa de Churn por variable categórica, Test Chi-cuadrado para significancia, análisis de variables clave (`Contract`, `tenure`). |
| `07_Churn_Preprocessing` | Conversión de `TotalCharges` a numérico, imputación de 11 nulos (clientes con tenure=0), codificación binaria Yes/No→1/0, OHE para `Contract`, `InternetService`, `PaymentMethod`, `tenure_group`; features derivadas `num_services`, `is_monthly_contract`, `no_value_added`; split 80/20 **estratificado** por Churn. |
| `08_Churn_Feature_Engineering` | Variance Threshold, Mutual Information (clasificación), Chi-cuadrado (SelectKBest), RF Importance con `class_weight='balanced'`, Logistic Regression L1. Ranking combinado normalizado con score final promedio. |
| `09_Churn_Model_Training` | Logistic Regression (GridSearch sobre `C`, `penalty`), KNN Classifier (curva K vs F1 + GridSearch), Random Forest (RandomizedSearch n_iter=30 + fine-tune GridSearch). Estrategia `class_weight='balanced'` en todos los modelos. |
| `10_Churn_Validation` | StratifiedKFold-5 y -10 CV, matrices de confusión comparativas, curvas ROC y Precision-Recall, learning curves, análisis de umbral óptimo para maximizar Recall sin sacrificar Precision. |

### Modelos entrenados
| Modelo | Hiperparámetro principal | Búsqueda |
|--------|--------------------------|----------|
| Logistic Regression (L2) | `C` ∈ {0.001 … 10}, `penalty` ∈ {l1, l2} | GridSearchCV |
| KNN Classifier | `k` (impar), `weights`, `metric` | GridSearchCV |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_*`, `max_features`, `class_weight` | RandomizedSearch + GridSearch |

**Desbalance:** `class_weight='balanced'` en todos los modelos  
**Métricas:** F1-Score · AUC-ROC · Precision · Recall · PR-AUC · Accuracy  
**Validación:** StratifiedKFold-5 y -10, análisis de umbral de decisión

---

## Dashboard Interactivo (Streamlit)

App unificada con selector de tarea (Regresión / Clasificación):

| Vista | Descripción |
|-------|-------------|
| Predicción Individual | Formulario interactivo con factores de riesgo en tiempo real |
| Predicción por Lote | Upload CSV + descarga de resultados |
| Dashboard Modelos | Métricas comparativas + CV con barras de error |
| Feature Importance | Gráfico de importancia RF (checklist del taller) |
| Análisis Dataset | Distribuciones, correlación, segmentación por variables clave |

### Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

### Desplegar en Streamlit Cloud
```
1. Haz fork de este repositorio
2. Ve a https://share.streamlit.io
3. Conecta tu repo → Main file: app/app.py
```

---

## Orden de Ejecución

```
1. Coloca los datasets en data/raw/
2. Ejecuta: 01 → 02 → 03 → 04 → 05   (Regresión — Medical Insurance)
3. Ejecuta: 06 → 07 → 08 → 09 → 10   (Clasificación — Telco Churn)
4. streamlit run app/app.py
```

---

## Checklist del Taller

- [x] Repositorio con carpetas `/data`, `/notebooks`, `/app`
- [x] 10 notebooks comentados con conclusiones por sección
- [x] `requirements.txt` completo para replicar el entorno
- [x] 3 modelos por tarea con búsqueda de hiperparámetros (GridSearch + RandomizedSearch)
- [x] EDA completo: correlaciones, valores faltantes, distribuciones, sesgo
- [x] Preprocesamiento: OHE + Label Encoding + StandardScaler (fit solo en train)
- [x] Ingeniería de características: 4–5 métodos de selección por tarea
- [x] Cross-Validation robusta (KFold / StratifiedKFold) + métricas en Test set
- [x] Dashboard con gráfico de Feature Importance (RF)
- [x] Predicción individual + predicción por lote (CSV)
- [x] Manejo de desbalance de clases: `class_weight='balanced'`
- [x] Transformación logarítmica del target en regresión (`log1p` / `expm1`)
- [x] Análisis de umbral óptimo para clasificación (Recall vs Precision)


