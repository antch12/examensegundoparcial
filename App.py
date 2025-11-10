import streamlit as st
import joblib
import pandas as pd

# Cargar modelo y nombres de características
modelo = joblib.load("lung_cancer_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Predicción de Cáncer de Pulmón", layout="centered")

st.title("Predicción de Cáncer de Pulmón")
st.write("Esta aplicación predice el riesgo de cáncer de pulmón basado en síntomas y factores de riesgo.")

st.markdown("---")

with st.form("formulario_cancer"):
    st.subheader("Información del Paciente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        genero = st.selectbox("Género:", ["Masculino", "Femenino"])
    
    with col2:
        edad = st.number_input("Edad:", min_value=20, max_value=90, value=60)
    
    st.subheader("Síntomas y Factores de Riesgo")
    st.write("Selecciona **Sí** si el paciente presenta el síntoma:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fumador = st.radio("Fumador:", ["No", "Sí"], horizontal=True)
        dedos_amarillos = st.radio("Dedos Amarillos:", ["No", "Sí"], horizontal=True)
        ansiedad = st.radio("Ansiedad:", ["No", "Sí"], horizontal=True)
        presion_pares = st.radio("Presión de Pares:", ["No", "Sí"], horizontal=True)
        enfermedad_cronica = st.radio("Enfermedad Crónica:", ["No", "Sí"], horizontal=True)
        fatiga = st.radio("Fatiga:", ["No", "Sí"], horizontal=True)
        alergia = st.radio("Alergia:", ["No", "Sí"], horizontal=True)
    
    with col2:
        sibilancias = st.radio("Sibilancias:", ["No", "Sí"], horizontal=True)
        alcohol = st.radio("Consumo de Alcohol:", ["No", "Sí"], horizontal=True)
        tos = st.radio("Tos:", ["No", "Sí"], horizontal=True)
        falta_aire = st.radio("Falta de Aire:", ["No", "Sí"], horizontal=True)
        dificultad_tragar = st.radio("Dificultad al Tragar:", ["No", "Sí"], horizontal=True)
        dolor_pecho = st.radio("Dolor de Pecho:", ["No", "Sí"], horizontal=True)
    
    submit = st.form_submit_button("🔍 Realizar Predicción", use_container_width=True)

if submit:
    genero_num = 1 if genero == "Masculino" else 0
    valores = []
    mapeo = {
        'GENDER': genero_num,
        'AGE': edad,
        'SMOKING': 1 if fumador == "Sí" else 0,
        'YELLOW_FINGERS': 1 if dedos_amarillos == "Sí" else 0,
        'ANXIETY': 1 if ansiedad == "Sí" else 0,
        'PEER_PRESSURE': 1 if presion_pares == "Sí" else 0,
        'CHRONIC DISEASE': 1 if enfermedad_cronica == "Sí" else 0,
        'FATIGUE ': 1 if fatiga == "Sí" else 0,
        'ALLERGY ': 1 if alergia == "Sí" else 0,
        'WHEEZING': 1 if sibilancias == "Sí" else 0,
        'ALCOHOL CONSUMING': 1 if alcohol == "Sí" else 0,
        'COUGHING': 1 if tos == "Sí" else 0,
        'SHORTNESS OF BREATH': 1 if falta_aire == "Sí" else 0,
        'SWALLOWING DIFFICULTY': 1 if dificultad_tragar == "Sí" else 0,
        'CHEST PAIN': 1 if dolor_pecho == "Sí" else 0
    }
    
    for feature in feature_names:
        valores.append(mapeo.get(feature, 0))
    
    entrada = pd.DataFrame([valores], columns=feature_names)
    with st.expander("Ver datos de entrada"):
        st.write(entrada)
    
    prediccion = modelo.predict(entrada)[0]
    probabilidad = modelo.predict_proba(entrada)[0]
    
    st.markdown("---")
    st.subheader("Resultado de la Predicción")
    
    if prediccion == 1:
        st.error("**RIESGO DETECTADO** - El modelo predice presencia de cáncer de pulmón")
        st.metric("Probabilidad de Cáncer", f"{probabilidad[1]*100:.1f}%")
    else:
        st.success("**SIN RIESGO APARENTE** - El modelo no detecta cáncer de pulmón")
        st.metric("Probabilidad de NO tener Cáncer", f"{probabilidad[0]*100:.1f}%")
    
    st.info("**Nota importante:** Este resultado es generado por un modelo de machine learning y debe ser validado por un profesional médico.")

st.markdown("---")
st.caption("Desarrollado con Streamlit y Logistic Regression | Modelo entrenado con SMOTE")