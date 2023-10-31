'''
Agrupamiento: Este es un método para dividir un conjunto de datos
en grupos homogéneos basados en sus características. Se puede usar
para encontrar patrones, como segmentar clientes según sus preferencias,
agrupar documentos según su temática o identificar anomalías. Un ejemplo
de código en Python es el siguiente:
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generar datos sintéticos
x1 = np.random.normal(2, 0.5, (50, 2)) # grupo 0
x2 = np.random.normal(4, 0.5, (50, 2)) # grupo 1
x3 = np.random.normal(6, 0.5, (50, 2)) # grupo 2
x = np.concatenate((x1, x2, x3)) # variable independiente

# Crear y entrenar el modelo de agrupamiento
model = KMeans(n_clusters=3)
model.fit(x)

# Obtener las etiquetas y los centroides del modelo
y = model.labels_ # etiquetas de los grupos
c = model.cluster_centers_ # coordenadas de los centroides

# Graficar los datos y los centroides del modelo
plt.scatter(x[:, 0], x[:, 1], c=y, cmap="rainbow")
plt.scatter(c[:, 0], c[:, 1], c="k", marker="x", s=100)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
