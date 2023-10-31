'''
Clasificación binaria: Este es un método para asignar una etiqueta
a una observación basada en sus características. Se puede usar para
clasificar objetos, como si una imagen contiene un gato o un perro,
si un correo electrónico es spam o no, o si un tumor es benigno o
maligno. Un ejemplo de código en Python es el siguiente:
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Generar datos sintéticos
x1 = np.random.normal(2, 1, (50, 2)) # clase 0
x2 = np.random.normal(4, 1, (50, 2)) # clase 1
# normal(valores al rededor, amplitud al rededor del primero valor, (cantidad de valores, numero de columnas))
x = np.concatenate((x1, x2)) # variable independiente
y = np.array([0] * 50 + [1] * 50) # variable dependiente

# Crear y entrenar el modelo de clasificación binaria
model = LogisticRegression()
model.fit(x, y)

# Obtener los coeficientes del modelo
a = model.intercept_ # término independiente
b = model.coef_ # vector de pesos

# Mostrar la ecuación del modelo
print(f"y = {a} + {b[0][0]}x1 + {b[0][1]}x2")

# Graficar los datos y la frontera de decisión del modelo
plt.scatter(x1[:, 0], x1[:, 1], c="b", label="Clase 0")
plt.scatter(x2[:, 0], x2[:, 1], c="r", label="Clase 1")
x1_min, x1_max = plt.xlim()
x2_min = (-a - b[0][0] * x1_min) / b[0][1]
x2_max = (-a - b[0][0] * x1_max) / b[0][1]
plt.plot([x1_min, x1_max], [x2_min, x2_max], "k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()
