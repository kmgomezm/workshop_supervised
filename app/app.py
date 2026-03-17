"""
ML Dashboard — Regresión & Clasificación · EAFIT 2026
Optimizado para Streamlit Cloud (memoria mínima)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, json, os, gc

st.set_page_config(
    page_title="ML · EAFIT 2026",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Rutas ─────────────────────────────────────────────────
def _root():
    c = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.isdir(os.path.join(c, 'models')): return c
        c = os.path.dirname(c)
    return os.getcwd()

ROOT   = _root()
MODELS = os.path.join(ROOT, 'models')
DATA   = os.path.join(ROOT, 'data', 'processed')
RAW    = os.path.join(ROOT, 'data', 'raw')

# ── Carga de modelos (una sola vez, compartida) ───────────
@st.cache_resource(show_spinner="Cargando modelos…")
def load_all_models():
    def _load(path):
        return joblib.load(path)
    ins, churn = {}, {}
    try:
        ins = {
            'Random Forest': _load(os.path.join(MODELS, 'random_forest.pkl')),
            'Reg. Lineal':   _load(os.path.join(MODELS, 'linear_regression.pkl')),
            'KNN':           _load(os.path.join(MODELS, 'knn_regressor.pkl')),
            '_scaler':       _load(os.path.join(MODELS, 'scaler.pkl')),
            '_feats':        _load(os.path.join(MODELS, 'selected_features.pkl')),
        }
    except Exception as e:
        st.warning(f"Insurance models: {e}")
    try:
        churn = {
            'Random Forest':   _load(os.path.join(MODELS, 'churn_rf.pkl')),
            'Log. Regression': _load(os.path.join(MODELS, 'churn_logistic.pkl')),
            'KNN':             _load(os.path.join(MODELS, 'churn_knn.pkl')),
            '_scaler':         _load(os.path.join(MODELS, 'churn_scaler.pkl')),
            '_feats':          _load(os.path.join(MODELS, 'churn_selected_features.pkl')),
        }
    except Exception as e:
        st.warning(f"Churn models: {e}")
    gc.collect()
    return ins, churn

@st.cache_data(show_spinner=False)
def load_report(fname):
    try:
        with open(os.path.join(DATA, fname)) as f:
            return json.load(f)
    except:
        return {}

@st.cache_data(show_spinner=False, max_entries=2)
def load_raw(name):
    """Carga CSV crudo y retorna muestra de 800 filas para EDA."""
    path = os.path.join(RAW, name)
    try:
        df = pd.read_csv(path)
        return df.sample(min(800, len(df)), random_state=42)
    except:
        return None

# ── Preprocessing ─────────────────────────────────────────
def prep_ins(row_dict, scaler, feats):
    d = pd.DataFrame([row_dict])
    # Normalizar texto
    for c in ['sex','smoker','region']:
        if c in d: d[c] = d[c].str.lower().str.strip()
    # Features de interacción (igual que Notebook 02)
    d['smoker_enc'] = (d['smoker'] == 'yes').astype(int)
    d['bmi_smoker'] = d['bmi'] * d['smoker_enc']
    d['age_smoker'] = d['age'] * d['smoker_enc']
    d['bmi_category'] = pd.cut(d['bmi'], bins=[0,18.5,24.9,29.9,100],
                                labels=['underweight','normal','overweight','obese'])
    d['age_group']    = pd.cut(d['age'], bins=[0,30,45,100],
                                labels=['young','middle','senior'])
    d = pd.get_dummies(d, columns=['sex','region','bmi_category','age_group'],
                       drop_first=False, dtype=int)
    # Scaler fue entrenado SOLO en estas 5 columnas (Notebook 02)
    num_cols = ['age','bmi','children','bmi_smoker','age_smoker']
    num_present = [c for c in num_cols if c in d.columns]
    d[num_present] = scaler.transform(d[num_present])
    # Ahora seleccionar features (algunas pueden haber sido eliminadas en NB03)
    for f in feats:
        if f not in d.columns: d[f] = 0
    Xs = d[feats].copy()   # scaled version
    # Para RF (no necesita scaling): reconstruir sin escalar
    d2 = pd.DataFrame([row_dict])
    for c in ['sex','smoker','region']:
        if c in d2: d2[c] = d2[c].str.lower().str.strip()
    d2['smoker_enc'] = (d2['smoker'] == 'yes').astype(int)
    d2['bmi_smoker'] = d2['bmi'] * d2['smoker_enc']
    d2['age_smoker'] = d2['age'] * d2['smoker_enc']
    d2['bmi_category'] = pd.cut(d2['bmi'], bins=[0,18.5,24.9,29.9,100],
                                 labels=['underweight','normal','overweight','obese'])
    d2['age_group']    = pd.cut(d2['age'], bins=[0,30,45,100],
                                 labels=['young','middle','senior'])
    d2 = pd.get_dummies(d2, columns=['sex','region','bmi_category','age_group'],
                        drop_first=False, dtype=int)
    for f in feats:
        if f not in d2.columns: d2[f] = 0
    Xr = d2[feats].copy()  # raw (unscaled) version
    return Xs, Xr

def _build_churn_df(row_dict):
    """Construye el DataFrame de features para churn (sin escalar)."""
    d = pd.DataFrame([row_dict])
    for c in d.select_dtypes('object').columns:
        d[c] = d[c].str.strip()
    d['TotalCharges'] = pd.to_numeric(d.get('TotalCharges', pd.Series([0])), errors='coerce').fillna(0)
    d['Partner_enc']          = (d.get('Partner',          pd.Series(['No'])) == 'Yes').astype(int)
    d['Dependents_enc']       = (d.get('Dependents',       pd.Series(['No'])) == 'Yes').astype(int)
    d['PhoneService_enc']     = (d.get('PhoneService',     pd.Series(['No'])) == 'Yes').astype(int)
    d['PaperlessBilling_enc'] = (d.get('PaperlessBilling', pd.Series(['No'])) == 'Yes').astype(int)
    d['gender_enc']           = (d.get('gender',           pd.Series(['Male'])) == 'Male').astype(int)
    tv = ['MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection',
          'TechSupport','StreamingTV','StreamingMovies']
    for c in tv:
        d[c+'_enc'] = (d.get(c, pd.Series(['No'])) == 'Yes').astype(int)
    ohe = ['Contract','InternetService','PaymentMethod']
    tenure = float(d['tenure'].iloc[0]) if 'tenure' in d.columns else 0
    mc     = float(d['MonthlyCharges'].iloc[0]) if 'MonthlyCharges' in d.columns else 0
    tc     = float(d['TotalCharges'].iloc[0])
    d['avg_monthly_charge'] = tc / tenure if tenure > 0 else mc
    d['num_services']        = sum(int((d.get(c, pd.Series(['No'])) == 'Yes').iloc[0]) for c in tv)
    d['is_monthly_contract'] = int(d.get('Contract', pd.Series([''])). iloc[0] == 'Month-to-month')
    d['is_new_customer']     = int(tenure <= 6)
    d['is_fiber_optic']      = int(d.get('InternetService', pd.Series([''])).iloc[0] == 'Fiber optic')
    d['no_value_added']      = int(
        d.get('OnlineSecurity', pd.Series(['No'])).iloc[0] == 'No' and
        d.get('TechSupport',    pd.Series(['No'])).iloc[0] == 'No' and
        d.get('OnlineBackup',   pd.Series(['No'])).iloc[0] == 'No'
    )
    d['tenure_group'] = pd.cut(pd.Series([tenure]), bins=[0,12,24,48,72],
        labels=['new_0_12','medium_12_24','loyal_24_48','champion_48_72'], include_lowest=True)
    ohe.append('tenure_group')
    d = pd.get_dummies(d, columns=[c for c in ohe if c in d.columns],
                       drop_first=False, dtype=int)
    return d

def prep_churn(row_dict, scaler, feats):
    d  = _build_churn_df(row_dict)
    for f in feats:
        if f not in d.columns: d[f] = 0
    Xr = d[feats].copy()   # unscaled — para RF
    Xs = Xr.copy()
    # Scaler fue entrenado SOLO en estas columnas (Notebook 07)
    num_cols = ['tenure','MonthlyCharges','TotalCharges','avg_monthly_charge','num_services']
    num_present = [c for c in num_cols if c in Xs.columns]
    Xs[num_present] = scaler.transform(Xs[num_present])
    return Xs, Xr

# ══════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════
st.markdown("## 🤖 ML Dashboard · Aprendizaje Supervisado · EAFIT 2026")
st.caption("Regresión: Medical Insurance  |  Clasificación: Telco Churn")
st.divider()

ins_models, churn_models = load_all_models()
ins_ok   = bool(ins_models)
churn_ok = bool(churn_models)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Configuración")
    task = st.radio("**Tarea:**", ["🏥 Regresión — Insurance", "📡 Clasificación — Churn"])
    is_reg = task.startswith("🏥")
    st.divider()
    nav = st.radio("**Vista:**", [
        "🎯 Predicción Individual",
        "📂 Predicción por Lote",
        "📊 Dashboard Modelos",
        "🔍 Feature Importance",
        "📈 EDA",
    ])
    st.divider()
    if is_reg:
        sel = st.selectbox("**Modelo activo:**",
                           [k for k in ins_models if not k.startswith('_')]) if ins_ok else None
    else:
        sel = st.selectbox("**Modelo activo:**",
                           [k for k in churn_models if not k.startswith('_')]) if churn_ok else None

# ══════════════════════════════════════════════════════════
# REGRESIÓN
# ══════════════════════════════════════════════════════════
if is_reg:
    if not ins_ok:
        st.error("⚠️ Ejecuta los notebooks 01-05 y sube los .pkl a /models")
        st.stop()

    scaler = ins_models['_scaler']
    feats  = ins_models['_feats']
    mnames = [k for k in ins_models if not k.startswith('_')]

    # ── Predicción Individual ─────────────────────────────
    if nav == "🎯 Predicción Individual":
        st.subheader("🏥 Predicción de Costo Médico")
        c1, c2 = st.columns(2)
        with c1:
            age      = st.slider("Edad", 18, 64, 35)
            sex      = st.selectbox("Género", ["male","female"])
            children = st.slider("Hijos cubiertos", 0, 5, 0)
            region   = st.selectbox("Región", ["northeast","northwest","southeast","southwest"])
        with c2:
            bmi    = st.slider("BMI", 10.0, 55.0, 26.5, 0.1)
            smoker = st.selectbox("¿Fumador?", ["no","yes"])
            bmi_lbl = ("🟢 Normal" if 18.5<=bmi<=24.9 else
                       "🔵 Bajo peso" if bmi<18.5 else
                       "🟡 Sobrepeso" if bmi<=29.9 else "🔴 Obeso")
            st.info(f"Categoría BMI: **{bmi_lbl}**")
            if smoker == "yes":
                st.warning("⚠️ Fumador: costo estimado 3-4× mayor")

        if st.button("🚀 Predecir", use_container_width=True):
            row = dict(age=age,sex=sex,bmi=bmi,children=children,smoker=smoker,region=region)
            Xs, Xr = prep_ins(row, scaler, feats)
            preds = {}
            for nm in mnames:
                Xi = Xs if nm in ['Reg. Lineal','KNN'] else Xr
                preds[nm] = float(np.expm1(ins_models[nm].predict(Xi)[0]))

            st.metric(f"💰 Costo estimado — {sel}", f"${preds[sel]:,.2f} USD/año")
            fig = go.Figure(go.Bar(
                x=list(preds.keys()), y=list(preds.values()),
                text=[f"${v:,.0f}" for v in preds.values()],
                textposition='outside',
                marker_color=['#6366f1','#0ea5e9','#22c55e']
            ))
            fig.update_layout(title="Comparación entre modelos",
                              yaxis_title="Costo (USD)", showlegend=False,
                              height=320, margin=dict(t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Factores detectados:**")
            if smoker=="yes": st.error("🚬 Tabaquismo — factor #1 de costo elevado")
            if bmi>30:        st.warning(f"⚖️ Obesidad (BMI {bmi:.1f})")
            if age>50:        st.info(f"📅 Edad {age} años — costos más altos en promedio")

    # ── Lote ─────────────────────────────────────────────
    elif nav == "📂 Predicción por Lote":
        st.subheader("📂 Predicción Masiva — Insurance")
        tpl = pd.DataFrame({'age':[25,45],'sex':['male','female'],
                            'bmi':[22.5,30.1],'children':[0,2],
                            'smoker':['no','yes'],'region':['northeast','southwest']})
        st.download_button("⬇️ Descargar template",
                           tpl.to_csv(index=False).encode(), "template_insurance.csv")
        f = st.file_uploader("📤 Subir CSV", type=['csv'])
        if f:
            df_up = pd.read_csv(f)
            st.write(f"✅ {len(df_up)} registros")
            if st.button("🚀 Predecir lote"):
                Xs, Xr = prep_ins(df_up.iloc[0].to_dict(), scaler, feats)  # warm up
                results = []
                for _, row in df_up.iterrows():
                    Xs_r, Xr_r = prep_ins(row.to_dict(), scaler, feats)
                    for nm in mnames:
                        Xi = Xs_r if nm in ['Reg. Lineal','KNN'] else Xr_r
                        df_up.loc[_, f'pred_{nm}'] = float(np.expm1(ins_models[nm].predict(Xi)[0]))
                mc = f'pred_{sel}'
                c1,c2,c3 = st.columns(3)
                c1.metric("Promedio", f"${df_up[mc].mean():,.2f}")
                c2.metric("Mínimo",   f"${df_up[mc].min():,.2f}")
                c3.metric("Máximo",   f"${df_up[mc].max():,.2f}")
                st.dataframe(df_up.round(2), use_container_width=True)
                st.download_button("⬇️ Descargar resultados",
                                   df_up.to_csv(index=False).encode(),
                                   "predicciones_insurance.csv")

    # ── Dashboard ─────────────────────────────────────────
    elif nav == "📊 Dashboard Modelos":
        st.subheader("📊 Métricas — Regresión Insurance")
        rp = load_report("final_report.json")
        if not rp:
            st.warning("Ejecuta Notebook 05 para generar el reporte.")
        else:
            df_r = pd.DataFrame(rp).T[['test_rmse','test_mae','test_r2','cv5_r2_mean']].round(4)
            df_r.columns = ['RMSE ($)','MAE ($)','R² Test','R² CV-5']
            st.dataframe(df_r, use_container_width=True)
            fig = go.Figure()
            colors = ['#6366f1','#0ea5e9','#22c55e']
            for i,(nm,row) in enumerate(df_r.iterrows()):
                fig.add_trace(go.Bar(name=nm, x=df_r.columns, y=row.values,
                                     marker_color=colors[i]))
            fig.update_layout(barmode='group', height=380,
                              title="Comparación métricas por modelo",
                              margin=dict(t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

    # ── Feature Importance ────────────────────────────────
    elif nav == "🔍 Feature Importance":
        st.subheader("🔍 Feature Importance — Random Forest (Insurance)")
        rf = ins_models.get('Random Forest')
        if rf and hasattr(rf, 'feature_importances_'):
            fi = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=True)
            fi_pct = (fi / fi.sum() * 100).round(2)
            fig = go.Figure(go.Bar(
                x=fi_pct.values, y=fi_pct.index, orientation='h',
                marker=dict(color=fi_pct.values, colorscale='RdYlGn', showscale=True),
                text=[f'{v:.1f}%' for v in fi_pct.values], textposition='outside'
            ))
            fig.update_layout(title='Feature Importance (Gini) — Insurance',
                              xaxis_title='Importancia (%)',
                              height=max(380, len(fi)*26),
                              margin=dict(l=180, t=50, b=20, r=60))
            st.plotly_chart(fig, use_container_width=True)

    # ── EDA ───────────────────────────────────────────────
    elif nav == "📈 EDA":
        st.subheader("📈 EDA — Medical Insurance")
        df = load_raw('insurance.csv')
        if df is None:
            st.warning("Coloca `insurance.csv` en `data/raw/`")
        else:
            t1, t2, t3 = st.tabs(["Distribución charges","Correlación","Segmentación"])
            with t1:
                fig = px.histogram(df, x='charges', color='smoker', nbins=30,
                    color_discrete_map={'yes':'#e11d48','no':'#6366f1'}, opacity=0.75)
                fig.update_layout(height=360, margin=dict(t=30,b=20))
                st.plotly_chart(fig, use_container_width=True)
            with t2:
                dc = df.copy()
                dc['smoker_n'] = (dc['smoker']=='yes').astype(int)
                dc['sex_n']    = (dc['sex']=='male').astype(int)
                corr = dc[['age','bmi','children','smoker_n','sex_n','charges']].corr().round(3)
                fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                                zmin=-1, zmax=1)
                fig.update_layout(height=380, margin=dict(t=30,b=20))
                st.plotly_chart(fig, use_container_width=True)
            with t3:
                agg = df.groupby('smoker')['charges'].median().reset_index()
                fig = px.bar(agg, x='smoker', y='charges', color='smoker',
                    color_discrete_map={'yes':'#e11d48','no':'#6366f1'},
                    title='Mediana de charges por smoker')
                fig.update_layout(height=340, showlegend=False, margin=dict(t=40,b=20))
                st.plotly_chart(fig, use_container_width=True)
            del df; gc.collect()

# ══════════════════════════════════════════════════════════
# CLASIFICACIÓN
# ══════════════════════════════════════════════════════════
else:
    if not churn_ok:
        st.error("⚠️ Ejecuta los notebooks 06-10 y sube los .pkl a /models")
        st.stop()

    scaler = churn_models['_scaler']
    feats  = churn_models['_feats']
    mnames = [k for k in churn_models if not k.startswith('_')]

    # ── Predicción Individual ─────────────────────────────
    if nav == "🎯 Predicción Individual":
        st.subheader("📡 Predicción de Churn")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Datos de cuenta**")
            tenure   = st.slider("Meses como cliente", 0, 72, 12)
            contract = st.selectbox("Contrato", ["Month-to-month","One year","Two year"])
            mcharges = st.slider("Cargo mensual ($)", 18.0, 120.0, 65.0, 0.5)
            paperless= st.selectbox("Factura electrónica", ["Yes","No"])
            payment  = st.selectbox("Pago", ["Electronic check","Mailed check",
                                              "Bank transfer (automatic)","Credit card (automatic)"])
        with c2:
            st.markdown("**Servicios**")
            internet = st.selectbox("Internet", ["Fiber optic","DSL","No"])
            security = st.selectbox("Seguridad online", ["No","Yes","No internet service"])
            backup   = st.selectbox("Backup online",    ["No","Yes","No internet service"])
            tech     = st.selectbox("Tech Support",     ["No","Yes","No internet service"])
            tv       = st.selectbox("Streaming TV",     ["No","Yes","No internet service"])
            movies   = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])
        with c3:
            st.markdown("**Demografía**")
            gender  = st.selectbox("Género", ["Male","Female"])
            senior  = st.selectbox("Adulto mayor", ["No","Yes"])
            partner = st.selectbox("Pareja", ["No","Yes"])
            depend  = st.selectbox("Dependientes", ["No","Yes"])
            phone   = st.selectbox("Teléfono", ["Yes","No"])

        if st.button("🚀 Predecir Churn", use_container_width=True):
            row = dict(
                gender=gender, SeniorCitizen=1 if senior=="Yes" else 0,
                Partner=partner, Dependents=depend, tenure=tenure,
                PhoneService=phone, MultipleLines='No',
                InternetService=internet, OnlineSecurity=security,
                OnlineBackup=backup, DeviceProtection='No',
                TechSupport=tech, StreamingTV=tv, StreamingMovies=movies,
                Contract=contract, PaperlessBilling=paperless,
                PaymentMethod=payment, MonthlyCharges=mcharges,
                TotalCharges=float(mcharges*tenure)
            )
            Xs, Xr = prep_churn(row, scaler, feats)
            preds = {}
            for nm in mnames:
                Xi = Xs if nm in ['Log. Regression','KNN'] else Xr
                preds[nm] = float(churn_models[nm].predict_proba(Xi)[0,1])

            prob = preds[sel]
            is_churn = prob >= 0.5
            if is_churn:
                st.error(f"⚠️ CHURN PROBABLE — Probabilidad: **{prob*100:.1f}%** ({sel})")
            else:
                st.success(f"✅ CLIENTE RETENIDO — Probabilidad de churn: **{prob*100:.1f}%** ({sel})")

            fig = go.Figure(go.Bar(
                x=list(preds.keys()),
                y=[v*100 for v in preds.values()],
                text=[f"{v*100:.1f}%" for v in preds.values()],
                textposition='outside',
                marker_color=['#f43f5e' if v>=0.5 else '#22c55e' for v in preds.values()]
            ))
            fig.add_hline(y=50, line_dash='dash', line_color='gray', annotation_text='50%')
            fig.update_layout(title="Probabilidad de Churn por modelo",
                              yaxis_title="%", yaxis_range=[0,110],
                              showlegend=False, height=320, margin=dict(t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Factores de riesgo:**")
            if contract=="Month-to-month": st.error("📄 Contrato mensual — mayor riesgo")
            if tenure<=6:                  st.error(f"🆕 Cliente nuevo ({tenure} meses)")
            if internet=="Fiber optic":    st.warning("📶 Fiber optic — alta tasa histórica de churn")
            if security=="No" and tech=="No": st.warning("🔓 Sin seguridad ni soporte técnico")

    # ── Lote ─────────────────────────────────────────────
    elif nav == "📂 Predicción por Lote":
        st.subheader("📂 Predicción Masiva — Churn")
        tpl = pd.DataFrame({
            'gender':['Male','Female'], 'SeniorCitizen':[0,1],
            'Partner':['No','Yes'], 'Dependents':['No','No'],
            'tenure':[3,48], 'PhoneService':['Yes','Yes'],
            'MultipleLines':['No','Yes'], 'InternetService':['Fiber optic','DSL'],
            'OnlineSecurity':['No','Yes'], 'OnlineBackup':['No','Yes'],
            'DeviceProtection':['No','Yes'], 'TechSupport':['No','No'],
            'StreamingTV':['No','Yes'], 'StreamingMovies':['No','Yes'],
            'Contract':['Month-to-month','Two year'],
            'PaperlessBilling':['Yes','No'],
            'PaymentMethod':['Electronic check','Bank transfer (automatic)'],
            'MonthlyCharges':[75.35,45.20], 'TotalCharges':[226.05,2168.10]
        })
        st.download_button("⬇️ Descargar template",
                           tpl.to_csv(index=False).encode(), "template_churn.csv")
        f = st.file_uploader("📤 Subir CSV de clientes", type=['csv'])
        if f:
            df_up = pd.read_csv(f)
            st.write(f"✅ {len(df_up)} clientes cargados")
            if st.button("🚀 Predecir Churn masivo"):
                for nm in mnames:
                    probs = []
                    for _, row in df_up.iterrows():
                        Xs_r, Xr_r = prep_churn(row.to_dict(), scaler, feats)
                        Xi = Xs_r if nm in ['Log. Regression','KNN'] else Xr_r
                        probs.append(float(churn_models[nm].predict_proba(Xi)[0,1]))
                    df_up[f'prob_churn_{nm}'] = probs
                    df_up[f'pred_churn_{nm}'] = (df_up[f'prob_churn_{nm}'] >= 0.5).astype(int)
                mc = f'prob_churn_{sel}'
                pc = f'pred_churn_{sel}'
                ch = (df_up[pc]==1).sum()
                c1,c2,c3 = st.columns(3)
                c1.metric("Total clientes", len(df_up))
                c2.metric("Churn predicho", ch, f"{ch/len(df_up)*100:.1f}%")
                c3.metric("Alto riesgo >70%", (df_up[mc]>0.7).sum())
                fig = px.histogram(df_up, x=mc, nbins=20,
                    title=f"Distribución de probabilidad — {sel}",
                    color_discrete_sequence=['#f43f5e'])
                fig.add_vline(x=0.5, line_dash='dash', line_color='gray')
                fig.update_layout(height=320, margin=dict(t=40,b=20))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_up.round(3), use_container_width=True)
                st.download_button("⬇️ Descargar resultados",
                                   df_up.to_csv(index=False).encode(),
                                   "churn_predicciones.csv")

    # ── Dashboard ─────────────────────────────────────────
    elif nav == "📊 Dashboard Modelos":
        st.subheader("📊 Métricas — Clasificación Churn")
        rp = load_report("churn_final_report.json")
        if not rp:
            st.warning("Ejecuta Notebook 10 para generar el reporte.")
        else:
            df_r = pd.DataFrame(rp).T[
                ['test_accuracy','test_f1','test_precision','test_recall','test_auc_roc']
            ].round(4)
            df_r.columns = ['Accuracy','F1','Precision','Recall','AUC-ROC']
            st.dataframe(df_r, use_container_width=True)

            fig = go.Figure()
            colors = ['#6366f1','#f43f5e','#22c55e']
            for i,(nm,row) in enumerate(df_r.iterrows()):
                fig.add_trace(go.Bar(name=nm, x=df_r.columns, y=row.values,
                                     marker_color=colors[i],
                                     text=[f'{v:.3f}' for v in row.values],
                                     textposition='outside'))
            fig.update_layout(barmode='group', height=400,
                              title="Comparación métricas Churn",
                              yaxis_range=[0,1.12],
                              margin=dict(t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**CV StratifiedKFold-5 — F1:**")
            cv_data = {k: v.get('cv5_f1_mean',0) for k,v in rp.items()}
            cv_std  = {k: v.get('cv5_f1_std',0)  for k,v in rp.items()}
            fig2 = go.Figure([go.Bar(
                x=list(cv_data.keys()), y=list(cv_data.values()),
                error_y=dict(type='data', array=list(cv_std.values())),
                text=[f"{v:.3f}" for v in cv_data.values()],
                textposition='outside',
                marker_color=colors
            )])
            fig2.update_layout(title='F1 CV ± std', height=320,
                               showlegend=False, margin=dict(t=40,b=20))
            st.plotly_chart(fig2, use_container_width=True)

    # ── Feature Importance ────────────────────────────────
    elif nav == "🔍 Feature Importance":
        st.subheader("🔍 Feature Importance — Random Forest (Churn)")
        rf = churn_models.get('Random Forest')
        if rf and hasattr(rf, 'feature_importances_'):
            fi = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=True)
            fi_pct = (fi / fi.sum() * 100).round(2)
            fig = go.Figure(go.Bar(
                x=fi_pct.values, y=fi_pct.index, orientation='h',
                marker=dict(color=fi_pct.values, colorscale='RdYlGn', showscale=True),
                text=[f'{v:.1f}%' for v in fi_pct.values], textposition='outside'
            ))
            fig.update_layout(title='Feature Importance (Gini) — Churn',
                              xaxis_title='Importancia (%)',
                              height=max(400, len(fi)*26),
                              margin=dict(l=200, t=50, b=20, r=80))
            st.plotly_chart(fig, use_container_width=True)

    # ── EDA ───────────────────────────────────────────────
    elif nav == "📈 EDA":
        st.subheader("📈 EDA — Telco Customer Churn")
        df = load_raw('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        if df is None:
            st.warning("Coloca el CSV de Churn en `data/raw/`")
        else:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
            t1, t2, t3, t4 = st.tabs(["Balance clases","Numéricas","Correlación","Churn por categoría"])
            with t1:
                vc = df['Churn'].value_counts().reset_index()
                vc.columns = ['Churn','count']
                fig = px.pie(vc, names='Churn', values='count',
                             color='Churn', color_discrete_map={'No':'#22c55e','Yes':'#f43f5e'},
                             hole=0.4)
                fig.update_layout(height=360, margin=dict(t=30,b=20))
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"⚠️ Desbalance: "
                        f"{df['Churn'].value_counts(normalize=True)['Yes']*100:.1f}% Churn — "
                        "usar F1/AUC, no Accuracy")
            with t2:
                col = st.selectbox("Variable:", ['tenure','MonthlyCharges','TotalCharges'])
                agg = df.groupby('Churn')[col].median().reset_index()
                fig = px.bar(agg, x='Churn', y=col, color='Churn',
                             color_discrete_map={'No':'#22c55e','Yes':'#f43f5e'},
                             title=f'Mediana de {col} por Churn')
                fig.update_layout(height=340, showlegend=False, margin=dict(t=40,b=20))
                st.plotly_chart(fig, use_container_width=True)
            with t3:
                dc = df.copy()
                dc['Churn_n'] = (dc['Churn']=='Yes').astype(int)
                corr = dc[['tenure','MonthlyCharges','TotalCharges','SeniorCitizen','Churn_n']].corr().round(3)
                fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                                zmin=-1, zmax=1)
                fig.update_layout(height=360, margin=dict(t=30,b=20))
                st.plotly_chart(fig, use_container_width=True)
            with t4:
                cat = st.selectbox("Variable:", ['Contract','InternetService','PaymentMethod','gender'])
                cr = df.groupby(cat)['Churn'].apply(
                    lambda x: (x=='Yes').mean()*100).round(1).reset_index()
                cr.columns = [cat, 'Churn Rate (%)']
                cr = cr.sort_values('Churn Rate (%)', ascending=True)
                fig = px.bar(cr, x='Churn Rate (%)', y=cat, orientation='h',
                             color='Churn Rate (%)', color_continuous_scale='RdYlGn_r',
                             text='Churn Rate (%)', title=f'Churn Rate por {cat}')
                fig.update_layout(height=360, showlegend=False, margin=dict(t=40,b=20,l=160))
                st.plotly_chart(fig, use_container_width=True)
            del df; gc.collect()

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption("🤖 ML Dashboard · EAFIT 2026 · Scikit-learn · Streamlit · Plotly")
