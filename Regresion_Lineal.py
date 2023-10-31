'''
Regresión lineal: Este es un método para modelar la relación entre
una variable dependiente y una o más variables independientes. Se
puede usar para predecir valores numéricos, como el precio de una
casa, el salario de una persona o el rendimiento de un estudiante.
Un ejemplo de código en Python es el siguiente:
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generar datos sintéticos
x = np.random.rand(100, 1) # variable independiente (lista de valores > 0 y < 1)
y = 3 + 50 * x + np.random.randn(100, 1) # variable dependiente con ruido

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(x, y)

# Obtener los coeficientes del modelo
a = model.intercept_ # término independiente
b = model.coef_ # pendiente

# Mostrar la ecuación del modelo
print(f"y = {a} + {b}x")

# Graficar los datos y la recta del modelo
plt.scatter(x, y)
plt.plot(x, a + b * x, "r")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
