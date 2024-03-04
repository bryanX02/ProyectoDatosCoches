# AUTORES:
# BRYAN XAVIER QUILUMBA FARINANGO
# JESÚS MARÍA RODRÍGUEZ GARCÍA
# PABLO MANUEL RODRÍGUEZ SOSA

# Este programa es el encargado de procesar los datos crudos para obtenerlos en un formato

# Cargamos los datos Json a un DataFrame

# Librerías necesarias
from datetime import datetime
import pandas as pd

df_resultado = pd.read_json("../adquisicion/datosCrudos.json", orient="index")

df_resultado['Marca'] = df_resultado['Modelo'].str.split().str[0]

# Eliminar la marca de la columna 'Modelo'
df_resultado['Modelo'] = df_resultado.apply(lambda row: row['Modelo'].replace(row['Marca'], ''), axis=1)
# Eliminar espacios adicionales al inicio de la columna 'Modelo'
df_resultado['Modelo'] = df_resultado['Modelo'].str.strip()

column_order = ['Marca'] + [col for col in df_resultado.columns if col != 'Marca']

df_resultado['Precio(€)'] = df_resultado['Precio(€)'].replace({'€': ''}, regex=True)
df_resultado['Precio(€)'] = df_resultado['Precio(€)'].replace({'\\.': ''}, regex=True).astype(int)
#df_resultado.rename(columns={'Precio': 'Precio(€)'}, inplace=True)

df_resultado['Primera matriculación'] = pd.to_datetime(df_resultado['Primera matriculación'], format='%d.%m.%Y', errors='coerce')

df_resultado['Kilometraje(Km)'] = df_resultado['Kilometraje(Km)'].replace({'km': ''}, regex=True)
df_resultado['Kilometraje(Km)'] = df_resultado['Kilometraje(Km)'].replace({'\\.': ''}, regex=True).astype(int)
#df_resultado.rename(columns={'Kilometraje': 'Kilometraje(KM)'}, inplace=True)

df_resultado['Transmisión'] = df_resultado['Transmisión'].replace({'Cambio tipo': ''}, regex=True)
df_resultado['Transmisión'] = df_resultado['Transmisión'].replace({'automático': 'automatico'}, regex=True)

df_resultado['Potencia(Cv)'] = df_resultado['Potencia(Cv)'].str.extract('(\d+)', expand=False).astype(int)
#df_resultado.rename(columns={'Potencia': 'Potencia(CV)'}, inplace=True)

df_resultado['Tracción'] = df_resultado['Tracción'].replace({'Tracción ': ''}, regex=True)
df_resultado['Tracción'] = df_resultado['Tracción'].replace({'total (4x4)': 'total'}, regex=True)



# Utiliza expresiones regulares para extraer el valor numérico
df_resultado['Cilindrada(Cc)'] = df_resultado['Cilindrada(Cc)'].str.extract('(\d+)', expand=False).astype(int)

df_resultado = df_resultado[column_order]

current_date = datetime.now()
df_resultado['Edad(Años)'] = (current_date - df_resultado['Primera matriculación']).dt.days // 365

# Puede ser necesario instalar las dependencias pyarrow y fastparquet
df_resultado.to_parquet("datosProcesados.parquet", index = 0)


