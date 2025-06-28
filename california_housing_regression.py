# Predicción de precios de casas con regresión lineal (California Housing)
# Autor: Tu Nombre

# 1) Importar librerías
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 2) Cargar datos
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name="MedHouseValue")

# 3) Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4) Crear modelo de regresión lineal
model = LinearRegression()

# 5) Entrenar modelo
model.fit(X_train, y_train)

# 6) Predecir
y_pred = model.predict(X_test)

# 7) Evaluar resultados
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error cuadrático medio (MSE): {mse:.2f}")
print(f"Coeficiente de determinación (R2): {r2:.2f}")

# 8) Visualizar resultados
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Precios reales")
plt.ylabel("Precios predichos")
plt.title("Regresión lineal: precios reales vs predichos (California Housing)")
plt.plot([y_test.min(), y_test.max()], [
         y_test.min(), y_test.max()], "k--", lw=2)
plt.show()
