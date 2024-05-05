import pandas as pd
from funciones import cargar_parquet_drive

ID_DATOS = '1N9ucOBFVTi1A-LOU1UyqmU_-pUwR40SY'

df = cargar_parquet_drive(ID_DATOS)
# Calcular el precio promedio por marca y modelo
precio_base = df.groupby(['Marca', 'Modelo'])['Precio(€)'].mean()


# Función para calcular el precio estimado
def estimar_precio(marca, modelo, edad, kilometraje, potencia, transmision):
    # Obtener el precio base
    try:
        precio = precio_base[marca, modelo]
    except KeyError:
        precio = df['Precio(€)'].mean()  # Usar el promedio general si no hay datos específicos para la marca y modelo

    # Ajustar por la edad del coche
    precio *= (1 - 0.1 * edad)

    # Ajustar por el kilometraje
    precio -= (kilometraje / 10000) * (0.01 * precio_base[marca, modelo])

    # Ajustar por la potencia
    if potencia > df['Potencia(Cv)'].quantile(0.75):
        precio *= 1.05

    # Ajustar por el tipo de transmisión
    if transmision == 'automática':
        precio *= 1.53

    return round(precio, 2)


# Ejemplo de uso
marca = 'BMW'
modelo = 'Serie 1'
edad = 5
kilometraje = 50000
potencia = 120
transmision = 'automática'

precio_estimado = estimar_precio(marca, modelo, edad, kilometraje, potencia, transmision)
print(f"El precio estimado para el {marca} {modelo} es de €{precio_estimado}")
