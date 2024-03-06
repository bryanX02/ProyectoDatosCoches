import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_parquet("../limpieza/datosProcesados.parquet")

# Análisis Descriptivo
# Resumen estadístico de las variables numéricas
print(data.describe())

# Conteo de valores únicos en variables categóricas
print(data.select_dtypes(include=["object"]).nunique())

# Preparación para el análisis exploratorio
# Seleccionar solo las columnas numéricas para calcular correlaciones
data_numeric = data.select_dtypes(include=[np.number])

# Calcular la matriz de correlación para las variables numéricas
correlation_matrix = data_numeric.corr(method='spearman')

# Visualización de la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center= 0, vmin=-1, vmax=1)
plt.title("Matriz de Correlación")
plt.savefig(
    "../exploracion/matriz_correlacion.png"
)  # Guardar la figura en la carpeta de exploracion
plt.show()  # Mostrar la figura
plt.close()

# Visualización de la relación entre la marca y el precio
plt.figure(figsize=(10, 6))
sns.boxplot(x="Precio(€)", y="Marca", hue="Marca", data=data, palette="coolwarm", legend=False)
plt.title("Precio por Marca")
plt.xticks(rotation=45)
plt.savefig(
    "../exploracion/precio_por_marca.png"
)  # Guardar la figura en la carpeta de exploracion
plt.show()
plt.close()

# Precio según Edad, tipo de Carburante y transmisión
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="Edad(Años)", y="Precio(€)", data=data, hue="Carburante", style="Transmisión"
)
plt.title("Precio según Edad y Tipo de Carburante")
plt.savefig(
    "../exploracion/precio_segun_edad_y_carburante.png"
)  # Guardar la figura en la carpeta de exploracion
plt.show()  # Mostrar la figura
plt.close()

# Observaciones descritas en la memoria de la segunda entrega del proyecto
