# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 13:47:23 2025

@author: Julio Cortez
"""

# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Trabajo Final de Cálculo", layout="wide")
st.title("📊 Trabajo Final de Cálculo")

# Cargar dataset
st.header("1. Cargar Dataset")
uploaded_file = st.file_uploader("Sube el archivo CSV del dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset cargado correctamente")
    st.write(df.head())

    # Limpieza
    df['Tiempo_Segundos'] = df['Tiempo_Segundos'].apply(lambda x: x if x >= 0 else None)
    df[['Tiempo_Segundos']] = SimpleImputer(strategy='median').fit_transform(df[['Tiempo_Segundos']])
    df[['Dificultad_Percibida']] = SimpleImputer(strategy='most_frequent').fit_transform(df[['Dificultad_Percibida']])

    # Codificación de variables categóricas
    le_dict = {}
    for col in ['Tipo_Problema', 'Dificultad_Percibida', 'Resuelto_Correctamente']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Definir X e y
    X = df.drop(['ID_Estudiante', 'Resuelto_Correctamente'], axis=1)
    y = df['Resuelto_Correctamente']

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Resultados
    st.header("2. Evaluación del Modelo")

    st.subheader("📄 Reporte de Clasificación")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("🧮 Matriz de Confusión")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Matriz de Confusión')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    st.pyplot(fig)

    st.subheader("📌 Importancia de las Variables")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(importances)
else:
    st.info("🔄 Esperando que subas un archivo CSV...")
