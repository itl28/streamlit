import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Regresión Lineal Simple", page_icon="📈", layout="centered")

# --- Título y descripción ---
st.title("🔧 Predicción con Regresión Lineal Simple")
st.write("Aplicación interactiva para entrenar un modelo de **regresión lineal** y visualizar predicciones.")
st.write("Selecciona la variable independiente (X) y la dependiente (Y).")

# --- Cargar datos ---
st.subheader("Cargar datos")
uploaded_file = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])

if uploaded_file is None:
    st.info("Sube un archivo CSV para continuar.")
    st.stop()

# Leer datos
data = pd.read_csv(uploaded_file)
st.write("**Vista previa de los datos**")
st.dataframe(data.head())

# Detectar columnas numéricas (pero permitimos elegir cualquier columna y convertimos)
cols = data.columns.tolist()
x_col = st.selectbox("Selecciona la variable independiente (X)", cols, index=0)
y_col = st.selectbox("Selecciona la variable dependiente (Y)", cols, index=min(1, len(cols)-1))

# --- Preparación de datos: conversión a numérico y limpieza ---
df = data[[x_col, y_col]].copy()
df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
n_before = len(df)
df = df.dropna(subset=[x_col, y_col])
n_after = len(df)

if n_after == 0:
    st.error("No hay datos numéricos válidos para esas columnas (todo quedó como NaN). Elige otras columnas o limpia el CSV.")
    st.stop()

if n_after < n_before:
    st.warning(f"Se descartaron {n_before - n_after} filas por valores no numéricos o vacíos en {x_col}/{y_col}.")

X = df[[x_col]].values  # 2D (n,1)
y = df[y_col].values    # 1D (n,)

# --- Entrenamiento del modelo ---
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
coef = model.coef_[0]
intercept = model.intercept_

st.subheader("Resultados del modelo")
st.write(f"**Ecuación:**  \n**{y_col} = {coef:.5f} · {x_col} + {intercept:.5f}**")
st.metric("R² (coeficiente de determinación)", f"{r2:.4f}")
st.write(f"**MSE (error cuadrático medio):** {mse:.4f}")

# --- Predicción interactiva ---
st.subheader("Predicción interactiva")
xmin, xmax = float(np.nanmin(X)), float(np.nanmax(X))
if xmin == xmax:
    xmin -= 1; xmax += 1

x_new = st.number_input(
    f"Ingrese un valor para {x_col}",
    value=float(np.round((xmin + xmax)/2, 3)),
    step=float(np.round((xmax - xmin)/100 if xmax>xmin else 1.0, 3))
)
y_new = model.predict(np.array([[x_new]]))[0]
st.success(f"Predicción: **{y_col} ≈ {y_new:.5f}** para **{x_col} = {x_new}**")

# --- Gráfica ---
st.subheader("Visualización")
fig = plt.figure(figsize=(8,5))
plt.scatter(X, y, label="Datos")
# Línea de regresión
x_line = np.linspace(xmin, xmax, 200).reshape(-1,1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, label="Regresión lineal")
# Punto predicho
plt.scatter([x_new], [y_new], marker="x", s=100, label="Predicción")
plt.xlabel(x_col); plt.ylabel(y_col); plt.legend(); plt.tight_layout()
st.pyplot(fig)

