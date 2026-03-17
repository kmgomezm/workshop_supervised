"""
╔══════════════════════════════════════════════════════════════════╗
║   🤖 ML Dashboard — Regresión & Clasificación                    ║
║   🏥 Medical Insurance  |  📡 Telco Customer Churn               ║
║   Maestría en Ciencia de Datos · EAFIT 2026                      ║
╚══════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, json, os

st.set_page_config(page_title="ML Supervisado · EAFIT", page_icon="🤖",
                   layout="wide", initial_sidebar_state="expanded")

BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, '..', 'models')
DATA   = os.path.join(BASE, '..', 'data', 'processed')

# ── CSS ──────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Fira+Code:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif!important}
.hdr{background:linear-gradient(135deg,#0a0a1a,#0d1b3e,#1a0a3e);padding:2.2rem 3rem;
     border-radius:18px;margin-bottom:1.2rem;border:1px solid rgba(99,102,241,.3)}
.hdr h1{color:#f8fafc;font-size:2.1rem;font-weight:800;margin:0;letter-spacing:-.8px}
.hdr p{color:#94a3b8;font-size:.9rem;margin:.3rem 0 0}
.badge{background:rgba(99,102,241,.2);border:1px solid rgba(99,102,241,.5);color:#a5b4fc;
       font-size:.72rem;font-weight:600;padding:.15rem .6rem;border-radius:20px;margin:.1rem;display:inline-block}
.pred-reg{background:linear-gradient(135deg,#0ea5e9,#6366f1);padding:1.8rem;border-radius:16px;
          text-align:center;color:white;box-shadow:0 8px 28px rgba(99,102,241,.3)}
.pred-yes{background:linear-gradient(135deg,#f43f5e,#f97316);padding:1.8rem;border-radius:16px;
          text-align:center;color:white;box-shadow:0 8px 28px rgba(244,63,94,.3)}
.pred-no{background:linear-gradient(135deg,#22c55e,#16a34a);padding:1.8rem;border-radius:16px;
         text-align:center;color:white;box-shadow:0 8px 28px rgba(34,197,94,.3)}
.pamount{font-size:2.8rem;font-weight:800;font-family:'Fira Code'}
.plabel{font-size:.82rem;opacity:.85;text-transform:uppercase;letter-spacing:.1em}
.ins{background:#f8fafc;border-left:4px solid #6366f1;padding:.65rem 1rem;
     border-radius:0 8px 8px 0;font-size:.86rem;color:#374151;margin:.35rem 0}
.ins-r{border-left-color:#f43f5e}.ins-g{border-left-color:#22c55e}
.stButton>button{border:none!important;border-radius:10px!important;font-weight:700!important;
                  width:100%!important;padding:.6rem 1.5rem!important}
</style>""", unsafe_allow_html=True)

# ── LOADERS ──────────────────────────────────────────────
@st.cache_resource
def load_ins():
    try:
        m = {'Random Forest':joblib.load(os.path.join(MODELS,'random_forest.pkl')),
             'Reg. Lineal':joblib.load(os.path.join(MODELS,'linear_regression.pkl')),
             'KNN':joblib.load(os.path.join(MODELS,'knn_regressor.pkl'))}
        return m,joblib.load(os.path.join(MODELS,'scaler.pkl')),joblib.load(os.path.join(MODELS,'selected_features.pkl'))
    except: return None,None,None

@st.cache_resource
def load_churn():
    try:
        m = {'Random Forest':joblib.load(os.path.join(MODELS,'churn_rf.pkl')),
             'Log. Regression':joblib.load(os.path.join(MODELS,'churn_logistic.pkl')),
             'KNN':joblib.load(os.path.join(MODELS,'churn_knn.pkl'))}
        return m,joblib.load(os.path.join(MODELS,'churn_scaler.pkl')),joblib.load(os.path.join(MODELS,'churn_selected_features.pkl'))
    except: return None,None,None

@st.cache_data
def load_json(f):
    try:
        with open(os.path.join(DATA,f)) as fp: return json.load(fp)
    except: return {}

# ── PREPROCESSING ─────────────────────────────────────────
def prep_ins(df, scaler, feats):
    d=df.copy()
    for c in ['sex','smoker','region']:
        if c in d: d[c]=d[c].str.lower().str.strip()
    d['smoker_enc']=(d['smoker']=='yes').astype(int)
    d['bmi_smoker']=d['bmi']*d['smoker_enc']
    d['age_smoker']=d['age']*d['smoker_enc']
    d['bmi_category']=pd.cut(d['bmi'],bins=[0,18.5,24.9,29.9,100],labels=['underweight','normal','overweight','obese'])
    d['age_group']=pd.cut(d['age'],bins=[0,30,45,100],labels=['young','middle','senior'])
    d=pd.get_dummies(d,columns=['sex','region','bmi_category','age_group'],drop_first=False,dtype=int)
    for f in feats:
        if f not in d.columns: d[f]=0
    X=d[feats].copy(); Xs=X.copy()
    num=[c for c in ['age','bmi','children','bmi_smoker','age_smoker'] if c in X.columns]
    Xs[num]=scaler.transform(X[num])
    return Xs,X

def prep_churn(df, scaler, feats):
    d=df.copy()
    for c in d.select_dtypes('object').columns: d[c]=d[c].str.strip()
    d['TotalCharges']=pd.to_numeric(d.get('TotalCharges',0),errors='coerce').fillna(0)
    d['Partner_enc']=(d.get('Partner','No')=='Yes').astype(int)
    d['Dependents_enc']=(d.get('Dependents','No')=='Yes').astype(int)
    d['PhoneService_enc']=(d.get('PhoneService','No')=='Yes').astype(int)
    d['PaperlessBilling_enc']=(d.get('PaperlessBilling','No')=='Yes').astype(int)
    d['gender_enc']=(d.get('gender','Male')=='Male').astype(int)
    tv=['MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    for c in tv: d[c+'_enc']=(d.get(c,'No')=='Yes').astype(int)
    ohe=['Contract','InternetService','PaymentMethod']
    if 'tenure' in d.columns:
        d['tenure_group']=pd.cut(d['tenure'],bins=[0,12,24,48,72],labels=['new_0_12','medium_12_24','loyal_24_48','champion_48_72'],include_lowest=True)
        d['avg_monthly_charge']=np.where(d['tenure']>0,d['TotalCharges']/d['tenure'],d.get('MonthlyCharges',0))
        d['num_services']=sum(d[c+'_enc'] for c in tv if c+'_enc' in d.columns)
        d['is_monthly_contract']=(d.get('Contract','')=='Month-to-month').astype(int)
        d['is_new_customer']=(d['tenure']<=6).astype(int)
        d['is_fiber_optic']=(d.get('InternetService','')=='Fiber optic').astype(int)
        d['no_value_added']=((d.get('OnlineSecurity','No')=='No')&(d.get('TechSupport','No')=='No')&(d.get('OnlineBackup','No')=='No')).astype(int)
        ohe.append('tenure_group')
    d=pd.get_dummies(d,columns=[c for c in ohe if c in d.columns],drop_first=False,dtype=int)
    for f in feats:
        if f not in d.columns: d[f]=0
    X=d[feats].copy(); Xs=X.copy()
    num=[c for c in ['tenure','MonthlyCharges','TotalCharges','avg_monthly_charge','num_services'] if c in X.columns]
    if num: Xs[num]=scaler.transform(X[num])
    return Xs,X

# ── HEADER ────────────────────────────────────────────────
st.markdown("""<div class="hdr">
<h1>🤖 ML Dashboard · Aprendizaje Supervisado</h1>
<p>Maestría en Ciencia de Datos · EAFIT 2026 · Docente: Jorge I. Padilla-Buriticá</p>
<div style="margin-top:.7rem">
<span class="badge">🏥 Regresión · Medical Insurance</span>
<span class="badge">📡 Clasificación · Telco Churn</span>
<span class="badge">3 modelos por tarea</span>
<span class="badge">CV + GridSearch + RMSE/F1/AUC</span>
</div></div>""",unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Panel de Control")
    task=st.radio("**Tarea:**",[" 🏥 Regresión — Insurance"," 📡 Clasificación — Churn"])
    is_reg=(task.startswith(" 🏥"))
    st.markdown("---")

    if is_reg:
        ins_m,ins_sc,ins_f=load_ins()
        avail=list(ins_m.keys()) if ins_m else []
        sel=st.selectbox("**Modelo:**",avail) if avail else None
        rep=load_json("final_report.json")
        if rep:
            st.markdown("**R² Test Set:**")
            for k,v in rep.items(): st.markdown(f"`{k[:11]}` **{v.get('test_r2',0):.3f}**")
    else:
        chr_m,chr_sc,chr_f=load_churn()
        avail=list(chr_m.keys()) if chr_m else []
        sel=st.selectbox("**Modelo:**",avail) if avail else None
        rep=load_json("churn_final_report.json")
        if rep:
            st.markdown("**F1 Test Set:**")
            for k,v in rep.items(): st.markdown(f"`{k[:12]}` **{v.get('test_f1',0):.3f}**")

    st.markdown("---")
    nav=st.radio("**Vista:**",["🎯 Predicción Individual","📂 Predicción por Lote",
                               "📊 Dashboard Modelos","🔍 Feature Importance","📈 Análisis Dataset"])


# ═══════════════════════════════════════
# REGRESIÓN
# ═══════════════════════════════════════
if is_reg:
    if ins_m is None:
        st.error("⚠️ Ejecuta Notebooks 01-05 para generar los modelos.")
        st.stop()

    # ── Predicción Individual
    if nav=="🎯 Predicción Individual":
        st.markdown("## 🏥 Predicción de Costo Médico")
        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown("#### 👤 Datos")
            age=st.slider("Edad",18,64,35)
            sex=st.selectbox("Género",["male","female"])
            children=st.slider("Hijos",0,5,0)
            region=st.selectbox("Región",["northeast","northwest","southeast","southwest"])
        with c2:
            st.markdown("#### 🏋️ Salud")
            bmi=st.slider("BMI",10.0,55.0,26.5,0.1)
            smoker=st.selectbox("¿Fumador?",["no","yes"])
            cat="🟢 Normal" if 18.5<=bmi<=24.9 else "🔵 Bajo" if bmi<18.5 else "🟡 Sobrepeso" if bmi<=29.9 else "🔴 Obeso"
            st.info(f"BMI: **{cat}**")
        with c3:
            st.markdown("#### ⚙️ Opciones")
            compare=st.checkbox("Comparar todos",True)
            risk=(40 if smoker=="yes" else 0)+(25 if bmi>30 else 0)+(20 if age>50 else 0)
            st.metric("Índice de Riesgo",f"{risk}/85","🔴 Alto" if risk>=60 else "🟡 Mod." if risk>=30 else "🟢 Bajo")

        if st.button("🚀 Predecir Costo Médico",use_container_width=True):
            inp=pd.DataFrame([dict(age=age,sex=sex,bmi=bmi,children=children,smoker=smoker,region=region)])
            Xs,Xr=prep_ins(inp,ins_sc,ins_f)
            preds={nm:float(np.expm1(m.predict(Xs if nm in ['Reg. Lineal','KNN'] else Xr)[0])) for nm,m in ins_m.items()}
            main=preds[sel]
            st.markdown(f'''<div class="pred-reg"><div class="plabel">{sel}</div>
            <div class="pamount">${main:,.2f}</div><div class="plabel">USD / año</div></div>''',unsafe_allow_html=True)
            if compare:
                cols=st.columns(3)
                for i,(nm,v) in enumerate(preds.items()):
                    with cols[i]: st.metric(nm,f"${v:,.2f}",f"${v-main:+,.0f}" if nm!=sel else "← sel.")
                fig=go.Figure([go.Bar(x=list(preds.keys()),y=list(preds.values()),
                    text=[f"${v:,.0f}" for v in preds.values()],textposition='outside',
                    marker_color=['#0ea5e9','#6366f1','#22c55e'])])
                fig.update_layout(title="Comparación de Modelos",yaxis_title="Costo ($)",
                    showlegend=False,height=300,plot_bgcolor='white',paper_bgcolor='white')
                st.plotly_chart(fig,use_container_width=True)
            st.markdown("### 💡 Factores")
            if smoker=="yes": st.markdown('<div class="ins ins-r">🚬 <b>Tabaquismo:</b> Principal factor — 3-4x mayor costo</div>',unsafe_allow_html=True)
            if bmi>30: st.markdown(f'<div class="ins ins-r">⚖️ <b>Obesidad:</b> BMI {bmi:.1f} incrementa riesgo</div>',unsafe_allow_html=True)
            if age>50: st.markdown(f'<div class="ins">📅 <b>Edad:</b> {age} años — costos más altos</div>',unsafe_allow_html=True)

    # ── Lote
    elif nav=="📂 Predicción por Lote":
        st.markdown("## 📂 Predicción Masiva — Insurance")
        tpl=pd.DataFrame({'age':[25,45,55],'sex':['male','female','male'],
            'bmi':[22.5,30.1,28.7],'children':[0,2,1],'smoker':['no','yes','no'],
            'region':['northeast','southwest','southeast']})
        st.download_button("⬇️ Template CSV",tpl.to_csv(index=False).encode(),"template_insurance.csv","text/csv")
        f=st.file_uploader("📤 Sube CSV",type=['csv'])
        if f:
            df_up=pd.read_csv(f); st.write(f"✅ {len(df_up)} registros")
            if st.button("🚀 Ejecutar"):
                Xs,Xr=prep_ins(df_up,ins_sc,ins_f)
                for nm,m in ins_m.items():
                    df_up[f'pred_{nm}']=np.expm1(m.predict(Xs if nm in ['Reg. Lineal','KNN'] else Xr))
                mc=f'pred_{sel}'; c1,c2,c3,c4=st.columns(4)
                c1.metric("Promedio",f"${df_up[mc].mean():,.2f}"); c2.metric("Mínimo",f"${df_up[mc].min():,.2f}")
                c3.metric("Máximo",f"${df_up[mc].max():,.2f}"); c4.metric("Mediana",f"${df_up[mc].median():,.2f}")
                st.dataframe(df_up.round(2),use_container_width=True)
                st.download_button("⬇️ Descargar",df_up.to_csv(index=False).encode(),"predicciones_insurance.csv","text/csv")

    # ── Dashboard
    elif nav=="📊 Dashboard Modelos":
        st.markdown("## 📊 Dashboard Regresión")
        rp=load_json("final_report.json")
        if rp:
            df_r=pd.DataFrame(rp).T[['test_rmse','test_mae','test_r2','cv5_r2_mean']].round(4)
            df_r.columns=['RMSE($)','MAE($)','R²(Test)','R²(CV)']
            def hl(s): return ['background:#dcfce7;font-weight:bold' if (v==s.min() if s.name in ['RMSE($)','MAE($)'] else v==s.max()) else '' for v in s]
            st.dataframe(df_r.style.apply(hl),use_container_width=True)
            fig=make_subplots(rows=2,cols=2,subplot_titles=['RMSE↓','MAE↓','R²Test↑','R²CV↑'])
            cls=['#0ea5e9','#e11d48','#22c55e']; lb=[True,True,False,False]
            for (r,c),col,low in zip([(1,1),(1,2),(2,1),(2,2)],['RMSE($)','MAE($)','R²(Test)','R²(CV)'],lb):
                if col in df_r.columns:
                    v=df_r[col].values; b=np.argmin(v) if low else np.argmax(v)
                    mc=[cls[i] if i==b else '#cbd5e1' for i in range(len(v))]
                    fig.add_trace(go.Bar(x=list(df_r.index),y=v,marker_color=mc,
                        text=[f'{x:.3f}' for x in v],textposition='outside',showlegend=False),row=r,col=c)
            fig.update_layout(height=480,plot_bgcolor='white',paper_bgcolor='white')
            st.plotly_chart(fig,use_container_width=True)

    # ── Feature Importance
    elif nav=="🔍 Feature Importance":
        st.markdown("## 🔍 Feature Importance — Insurance")
        rf=ins_m.get('Random Forest')
        if rf and hasattr(rf,'feature_importances_'):
            fi=pd.Series(rf.feature_importances_,index=ins_f).sort_values(ascending=False)
            fi_p=fi/fi.sum()*100
            c1,c2=st.columns([3,2])
            with c1:
                fig=go.Figure(go.Bar(x=fi_p.values,y=fi_p.index,orientation='h',
                    marker=dict(color=fi_p.values,colorscale='RdYlGn',showscale=True),
                    text=[f'{v:.1f}%' for v in fi_p.values],textposition='outside'))
                fig.update_layout(title='Feature Importance (RF)',xaxis_title='%',
                    height=max(400,len(fi)*28),plot_bgcolor='white',paper_bgcolor='white',
                    yaxis=dict(autorange='reversed'))
                st.plotly_chart(fig,use_container_width=True)
            with c2:
                for feat,imp in fi_p.items():
                    w=int(imp/fi_p.max()*100); cl='#22c55e' if imp>fi_p.median() else '#94a3b8'
                    st.markdown(f'''<div style="margin:5px 0"><small style="font-weight:600">{feat}</small>
                    <div style="background:#f1f5f9;border-radius:4px"><div style="background:{cl};width:{w}%;height:9px;border-radius:4px"></div></div>
                    <small style="color:#64748b">{imp:.1f}%</small></div>''',unsafe_allow_html=True)

    # ── Dataset
    elif nav=="📈 Análisis Dataset":
        st.markdown("## 📈 EDA — Medical Insurance")
        try:
            df_r=pd.read_csv(os.path.join(BASE,'../data/raw/insurance.csv'))
            t1,t2,t3=st.tabs(["Distribuciones","Correlación","Segmentación"])
            with t1:
                col=st.selectbox("Variable:",["charges","age","bmi","children"])
                fig=px.histogram(df_r,x=col,color='smoker' if col=='charges' else None,nbins=40,
                    color_discrete_map={'yes':'#e11d48','no':'#0ea5e9'})
                fig.update_layout(plot_bgcolor='white',paper_bgcolor='white'); st.plotly_chart(fig,use_container_width=True)
            with t2:
                dc=df_r.copy(); dc['sm']=(dc['smoker']=='yes').astype(int); dc['sx']=(dc['sex']=='male').astype(int)
                corr=dc[['age','bmi','children','sm','sx','charges']].corr()
                fig=px.imshow(corr,text_auto=True,color_continuous_scale='RdBu_r'); st.plotly_chart(fig,use_container_width=True)
            with t3:
                fig=px.box(df_r,x='smoker',y='charges',color='smoker',
                    color_discrete_map={'yes':'#e11d48','no':'#0ea5e9'})
                fig.update_layout(plot_bgcolor='white',paper_bgcolor='white'); st.plotly_chart(fig,use_container_width=True)
        except: st.warning("Coloca insurance.csv en data/raw/")


# ═══════════════════════════════════════
# CLASIFICACIÓN
# ═══════════════════════════════════════
else:
    if chr_m is None:
        st.error("⚠️ Ejecuta Notebooks 06-10 para generar los modelos de clasificación.")
        st.stop()

    # ── Predicción Individual
    if nav=="🎯 Predicción Individual":
        st.markdown("## 📡 Predicción de Churn de Cliente")
        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown("#### 👤 Demográficos")
            gender=st.selectbox("Género",["Male","Female"])
            senior=st.selectbox("Adulto mayor",["No","Yes"])
            partner=st.selectbox("Pareja",["No","Yes"])
            depend=st.selectbox("Dependientes",["No","Yes"])
        with c2:
            st.markdown("#### 📋 Cuenta")
            tenure=st.slider("Meses como cliente",0,72,12)
            contract=st.selectbox("Contrato",["Month-to-month","One year","Two year"])
            mcharges=st.slider("Cargo mensual ($)",18.0,120.0,65.0,0.5)
            tcharges=float(mcharges*tenure)
            paperless=st.selectbox("Factura electrónica",["Yes","No"])
            payment=st.selectbox("Pago",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        with c3:
            st.markdown("#### 📶 Servicios")
            phone=st.selectbox("Teléfono",["Yes","No"])
            internet=st.selectbox("Internet",["Fiber optic","DSL","No"])
            security=st.selectbox("Seguridad",["No","Yes","No internet service"])
            backup=st.selectbox("Backup",["No","Yes","No internet service"])
            tech=st.selectbox("Tech Support",["No","Yes","No internet service"])
            tv=st.selectbox("Streaming TV",["No","Yes","No internet service"])
            movies=st.selectbox("Streaming Movies",["No","Yes","No internet service"])

        if st.button("🚀 Predecir Churn",use_container_width=True):
            inp=pd.DataFrame([{'gender':gender,'SeniorCitizen':1 if senior=="Yes" else 0,
                'Partner':partner,'Dependents':depend,'tenure':tenure,'PhoneService':phone,
                'MultipleLines':'No','InternetService':internet,'OnlineSecurity':security,
                'OnlineBackup':backup,'DeviceProtection':'No','TechSupport':tech,
                'StreamingTV':tv,'StreamingMovies':movies,'Contract':contract,
                'PaperlessBilling':paperless,'PaymentMethod':payment,
                'MonthlyCharges':mcharges,'TotalCharges':tcharges}])
            Xs,Xr=prep_churn(inp,chr_sc,chr_f)
            preds={nm:{'prob':m.predict_proba(Xs if nm in ['Log. Regression','KNN'] else Xr)[0,1],
                       'pred':m.predict(Xs if nm in ['Log. Regression','KNN'] else Xr)[0]} for nm,m in chr_m.items()}
            main=preds[sel]; prob=main['prob']; is_c=(main['pred']==1)
            box_c="pred-yes" if is_c else "pred-no"
            lbl="⚠️ CHURN DETECTADO" if is_c else "✅ CLIENTE RETENIDO"
            st.markdown(f'''<div class="{box_c}"><div class="plabel">{sel}</div>
            <div class="pamount">{lbl}</div>
            <div class="plabel">Probabilidad de Churn: {prob*100:.1f}%</div></div>''',unsafe_allow_html=True)
            st.markdown("### 📊 Probabilidad por Modelo")
            for nm,v in preds.items():
                p=v['prob']; cl='#f43f5e' if p>.5 else '#f97316' if p>.35 else '#22c55e'
                w=int(p*100)
                st.markdown(f'''<div style="margin:8px 0">
                <div style="display:flex;justify-content:space-between"><span style="font-weight:600">{nm}</span>
                <span style="font-family:monospace;color:{cl};font-weight:700">{p*100:.1f}%</span></div>
                <div style="background:#f1f5f9;border-radius:8px"><div style="background:{cl};width:{w}%;height:16px;border-radius:8px"></div></div>
                </div>''',unsafe_allow_html=True)
            st.markdown("### 💡 Factores de Riesgo")
            risks=[]
            if contract=="Month-to-month": risks.append("🔴 Contrato mensual — mayor riesgo de churn")
            if tenure<=6: risks.append(f"🔴 Cliente nuevo ({tenure} meses) — alta tasa de abandono")
            if internet=="Fiber optic": risks.append("🟠 Fiber optic — alta tasa de churn histórica")
            if security=="No" and tech=="No": risks.append("🟠 Sin seguridad ni soporte técnico")
            for r in risks: st.markdown(f'<div class="ins ins-r">{r}</div>',unsafe_allow_html=True)
            if not risks: st.markdown('<div class="ins ins-g">✅ No se detectaron factores de riesgo mayores</div>',unsafe_allow_html=True)

    # ── Lote
    elif nav=="📂 Predicción por Lote":
        st.markdown("## 📂 Predicción Masiva — Churn")
        tpl=pd.DataFrame({'gender':['Male','Female'],'SeniorCitizen':[0,1],'Partner':['No','Yes'],
            'Dependents':['No','No'],'tenure':[3,48],'PhoneService':['Yes','Yes'],
            'MultipleLines':['No','Yes'],'InternetService':['Fiber optic','DSL'],
            'OnlineSecurity':['No','Yes'],'OnlineBackup':['No','Yes'],
            'DeviceProtection':['No','Yes'],'TechSupport':['No','No'],
            'StreamingTV':['No','Yes'],'StreamingMovies':['No','Yes'],
            'Contract':['Month-to-month','Two year'],'PaperlessBilling':['Yes','No'],
            'PaymentMethod':['Electronic check','Bank transfer (automatic)'],
            'MonthlyCharges':[75.35,45.20],'TotalCharges':[226.05,2168.10]})
        st.download_button("⬇️ Template CSV Churn",tpl.to_csv(index=False).encode(),"template_churn.csv","text/csv")
        f=st.file_uploader("📤 Sube CSV Churn",type=['csv'])
        if f:
            df_up=pd.read_csv(f); st.write(f"✅ {len(df_up)} clientes")
            if st.button("🚀 Ejecutar predicciones Churn"):
                Xs,Xr=prep_churn(df_up,chr_sc,chr_f)
                for nm,m in chr_m.items():
                    Xi=Xs if nm in ['Log. Regression','KNN'] else Xr
                    df_up[f'prob_{nm}']=m.predict_proba(Xi)[:,1]
                    df_up[f'pred_{nm}']=m.predict(Xi)
                mc=f'prob_{sel}'; pc=f'pred_{sel}'
                ch=(df_up[pc]==1).sum(); c1,c2,c3,c4=st.columns(4)
                c1.metric("Total",len(df_up)); c2.metric("Churn",ch,f"{ch/len(df_up)*100:.1f}%")
                c3.metric("Prob. media",f"{df_up[mc].mean()*100:.1f}%"); c4.metric("Alto riesgo >70%",(df_up[mc]>.7).sum())
                fig=px.histogram(df_up,x=mc,nbins=25,title=f"Distribución Probabilidad Churn ({sel})",
                    color_discrete_sequence=['#f43f5e'])
                fig.add_vline(x=.5,line_dash='dash',line_color='gray')
                fig.update_layout(plot_bgcolor='white',paper_bgcolor='white'); st.plotly_chart(fig,use_container_width=True)
                st.dataframe(df_up.round(3),use_container_width=True)
                st.download_button("⬇️ Descargar",df_up.to_csv(index=False).encode(),"churn_predicciones.csv","text/csv")

    # ── Dashboard
    elif nav=="📊 Dashboard Modelos":
        st.markdown("## 📊 Dashboard Clasificación Churn")
        rp=load_json("churn_final_report.json")
        if rp:
            df_r=pd.DataFrame(rp).T[['test_accuracy','test_f1','test_precision','test_recall','test_auc_roc']].round(4)
            df_r.columns=['Accuracy','F1','Precision','Recall','AUC-ROC']
            def hl(s): return ['background:#dcfce7;font-weight:bold' if v==s.max() else '' for v in s]
            st.dataframe(df_r.style.apply(hl),use_container_width=True)
            fig=make_subplots(rows=2,cols=3,subplot_titles=['Accuracy','F1','Precision','Recall','AUC-ROC','CV F1'])
            cls=['#0ea5e9','#e11d48','#22c55e']
            mets=['Accuracy','F1','Precision','Recall','AUC-ROC']; pos=[(1,1),(1,2),(1,3),(2,1),(2,2)]
            for (r,c),m in zip(pos,mets):
                if m in df_r.columns:
                    v=df_r[m].values; b=np.argmax(v); mc=[cls[i] if i==b else '#cbd5e1' for i in range(len(v))]
                    fig.add_trace(go.Bar(x=list(df_r.index),y=v,marker_color=mc,
                        text=[f'{x:.3f}' for x in v],textposition='outside',showlegend=False),row=r,col=c)
            # CV F1
            if rp:
                cv_means=[rp[k]['cv5_f1_mean'] for k in rp]; cv_std=[rp[k].get('cv5_f1_std',0) for k in rp]
                fig.add_trace(go.Bar(x=list(rp.keys()),y=cv_means,
                    error_y=dict(type='data',array=cv_std),
                    marker_color=cls,text=[f'{v:.3f}' for v in cv_means],textposition='outside',showlegend=False),row=2,col=3)
            fig.update_layout(height=550,plot_bgcolor='white',paper_bgcolor='white')
            st.plotly_chart(fig,use_container_width=True)

    # ── Feature Importance
    elif nav=="🔍 Feature Importance":
        st.markdown("## 🔍 Feature Importance — Churn")
        rf=chr_m.get('Random Forest')
        if rf and hasattr(rf,'feature_importances_'):
            fi=pd.Series(rf.feature_importances_,index=chr_f).sort_values(ascending=False)
            fi_p=fi/fi.sum()*100
            c1,c2=st.columns([3,2])
            with c1:
                fig=go.Figure(go.Bar(x=fi_p.values,y=fi_p.index,orientation='h',
                    marker=dict(color=fi_p.values,colorscale='RdYlGn',showscale=True),
                    text=[f'{v:.1f}%' for v in fi_p.values],textposition='outside'))
                fig.update_layout(title='Feature Importance (RF Churn)',xaxis_title='%',
                    height=max(430,len(fi)*26),plot_bgcolor='white',paper_bgcolor='white',
                    yaxis=dict(autorange='reversed'))
                st.plotly_chart(fig,use_container_width=True)
            with c2:
                st.markdown("**Top Insights:**")
                for feat,imp in fi_p.head(5).items():
                    st.markdown(f'<div class="ins">🔑 <b>{feat}</b>: {imp:.1f}%</div>',unsafe_allow_html=True)
                st.markdown("---")
                for feat,imp in fi_p.items():
                    w=int(imp/fi_p.max()*100); cl='#f43f5e' if imp>fi_p.quantile(.75) else '#f97316' if imp>fi_p.median() else '#94a3b8'
                    st.markdown(f'''<div style="margin:4px 0"><div style="display:flex;justify-content:space-between">
                    <small style="font-weight:600;font-size:.73rem">{feat[:32]}</small>
                    <small style="color:{cl};font-family:monospace">{imp:.1f}%</small></div>
                    <div style="background:#f1f5f9;border-radius:4px"><div style="background:{cl};width:{w}%;height:7px;border-radius:4px"></div></div></div>''',unsafe_allow_html=True)

    # ── Dataset
    elif nav=="📈 Análisis Dataset":
        st.markdown("## 📈 EDA — Telco Churn")
        try:
            df_r=pd.read_csv(os.path.join(BASE,'../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'))
            df_r['TotalCharges']=pd.to_numeric(df_r['TotalCharges'],errors='coerce').fillna(0)
            t1,t2,t3,t4=st.tabs(["Churn Rate","Numéricas","Correlación","Segmentos"])
            with t1:
                vc=df_r['Churn'].value_counts()
                fig=go.Figure(go.Pie(labels=vc.index,values=vc.values,
                    marker_colors=['#22c55e','#f43f5e'],pull=[0,.08]))
                fig.update_layout(title='Distribución Churn',height=370); st.plotly_chart(fig,use_container_width=True)
                st.info(f"⚠️ Desbalance: {vc['Yes']/len(df_r)*100:.1f}% Churn vs {vc['No']/len(df_r)*100:.1f}% No Churn — usar F1/AUC, no Accuracy")
            with t2:
                col=st.selectbox("Variable:",["tenure","MonthlyCharges","TotalCharges"])
                fig=px.histogram(df_r,x=col,color='Churn',nbins=35,opacity=.7,
                    color_discrete_map={'No':'#22c55e','Yes':'#f43f5e'})
                fig.update_layout(plot_bgcolor='white',paper_bgcolor='white'); st.plotly_chart(fig,use_container_width=True)
            with t3:
                dc=df_r.copy(); dc['Churn_n']=(dc['Churn']=='Yes').astype(int)
                corr=dc[['tenure','MonthlyCharges','TotalCharges','SeniorCitizen','Churn_n']].corr()
                fig=px.imshow(corr,text_auto=True,color_continuous_scale='RdBu_r'); st.plotly_chart(fig,use_container_width=True)
            with t4:
                cat=st.selectbox("Variable:",["Contract","InternetService","PaymentMethod","gender"])
                cr=df_r.groupby(cat)['Churn'].apply(lambda x:(x=='Yes').mean()*100).sort_values()
                fig=go.Figure(go.Bar(x=cr.values,y=cr.index,orientation='h',
                    marker_color=['#22c55e' if v<20 else '#f97316' if v<35 else '#f43f5e' for v in cr.values],
                    text=[f'{v:.1f}%' for v in cr.values],textposition='outside'))
                fig.update_layout(title=f'Churn Rate por {cat}',xaxis_title='Churn Rate (%)',
                    height=370,plot_bgcolor='white',paper_bgcolor='white'); st.plotly_chart(fig,use_container_width=True)
        except: st.warning("Coloca WA_Fn-UseC_-Telco-Customer-Churn.csv en data/raw/")

# ── Footer
st.markdown("---")
st.markdown("""<div style="text-align:center;color:#94a3b8;font-size:.78rem;padding:.6rem 0">
🤖 <b>ML Dashboard · Aprendizaje Supervisado</b> · EAFIT 2026 ·
🏥 Medical Insurance (Regresión) · 📡 Telco Churn (Clasificación) · Scikit-learn · Streamlit · Plotly
</div>""",unsafe_allow_html=True)
