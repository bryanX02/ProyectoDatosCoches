from funciones import cargar_parquet_drive
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


# Carga de datos
ID_DATOS = '1N9ucOBFVTi1A-LOU1UyqmU_-pUwR40SY'
train_df = cargar_parquet_drive(ID_DATOS)
test_df = pd.read_parquet('datosNuevos.parquet')

# Calcular el precio base para cada marca y modelo en el conjunto de entrenamiento
precio_base = train_df.groupby(['Marca', 'Modelo'])['Precio(€)'].mean().to_dict()


# Función para calcular el precio estimado
def estimar_precio(row, precio_base):
    # Obtener el precio base
    clave = (row['Marca'], row['Modelo'])
    if clave in precio_base:
        precio = precio_base[clave]
    else:
        precio = train_df['Precio(€)'].mean()

    # Ajustar por la edad del coche
    precio *= (1 - 0.019 * row['Edad(Meses)'] / 12)

    # Ajustar por el kilometraje
    precio -= (row['Kilometraje(Km)'] / 10000) * (0.006 * precio)

    # Ajustar por la potencia
    if row['Potencia(Cv)'] > train_df['Potencia(Cv)'].quantile(0.75):
        precio *= 1.20

    # Ajustar por el tipo de transmisión
    if row['Transmisión'] == 'automática':
        precio *= 1.07

    return round(precio, 2)


# Aplicar la función al conjunto de pruebas
test_df['Precio Estimado'] = test_df.apply(lambda row: estimar_precio(row, precio_base), axis=1)

# Comparar resultados
print(test_df[['Marca', 'Modelo', 'Precio(€)', 'Precio Estimado']])

# Calcular Mae
mae = mean_absolute_error(test_df['Precio(€)'], test_df['Precio Estimado'])
print(f"Error Absoluto Medio (MAE): €{mae:.2f}")
