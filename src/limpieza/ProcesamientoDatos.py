# AUTORES:
# BRYAN XAVIER QUILUMBA FARINANGO
# JESÚS MARÍA RODRÍGUEZ GARCÍA
# PABLO MANUEL RODRÍGUEZ SOSA

# Este programa es el encargado de procesar los datos crudos para obtenerlos en un formato

# Cargamos los datos Json a un DataFrame

# Librerías necesarias
from datetime import datetime
import pandas as pd


# Función que le aplica los procedimientos de limpieza a los datos
def limpieza(df_resultado):

    #Creamos la columna Marca extrayendo la primera palabra de la columna Modelo
    df_resultado['Marca'] = df_resultado['Modelo'].str.split().str[0]

    # Eliminar la marca de la columna 'Modelo'
    df_resultado['Modelo'] = df_resultado.apply(lambda row: row['Modelo'].replace(row['Marca'], ''), axis=1)
    # Eliminar espacios adicionales al inicio de la columna 'Modelo'
    df_resultado['Modelo'] = df_resultado['Modelo'].str.strip()
    #ponemos a 'Marca' como la primera columna
    column_order = ['Marca'] + [col for col in df_resultado.columns if col != 'Marca']
    #quitamos el simbolo de euro y lo pasamos a entero
    df_resultado['Precio(€)'] = df_resultado['Precio(€)'].replace({'€': ''}, regex=True)
    df_resultado['Precio(€)'] = df_resultado['Precio(€)'].replace({'\\.': ''}, regex=True).astype(int)
    #df_resultado.rename(columns={'Precio': 'Precio(€)'}, inplace=True)
    #pasamos la primera matriculación a formato fecha
    df_resultado['Primera matriculación'] = pd.to_datetime(df_resultado['Primera matriculación'], format='%d.%m.%Y', errors='coerce')
    #quitamos el simbolo 'km' y lo pasamos a entero
    df_resultado['Kilometraje(Km)'] = df_resultado['Kilometraje(Km)'].replace({'km': ''}, regex=True)
    df_resultado['Kilometraje(Km)'] = df_resultado['Kilometraje(Km)'].replace({'\\.': ''}, regex=True).astype(int)
    #df_resultado.rename(columns={'Kilometraje': 'Kilometraje(KM)'}, inplace=True)
    #resucimos de 'Cambio tipo ...' a simplemente el tipo de cambio
    df_resultado['Transmisión'] = df_resultado['Transmisión'].replace({'Cambio tipo': ''}, regex=True)
    df_resultado['Transmisión'] = df_resultado['Transmisión'].replace({'automático': 'automatico'}, regex=True)
    #Dejamos solo el valor de los caballos y lo pasamos a entero
    df_resultado['Potencia(Cv)'] = df_resultado['Potencia(Cv)'].str.extract('(\d+)', expand=False).astype(int)
    #df_resultado.rename(columns={'Potencia': 'Potencia(CV)'}, inplace=True)
    #quitamos la palabra tracción para acortar
    df_resultado['Tracción'] = df_resultado['Tracción'].replace({'Tracción ': ''}, regex=True)
    df_resultado['Tracción'] = df_resultado['Tracción'].replace({'total (4x4)': 'total'}, regex=True)

    # Quitamos el símbolo 'ccm' y lo pasamos a entero
    df_resultado['Cilindrada(Cc)'] = df_resultado['Cilindrada(Cc)'].str.extract('(\d+)', expand=False).astype(int)

    df_resultado = df_resultado[column_order]
    #a partir de la columna primera matriculación, creamos la columna Edad que representa los años de antigüedad del vehículo teniendo en cuenta el año actual
    current_date = datetime.now()
    df_resultado['Edad(Meses)'] = (current_date - df_resultado['Primera matriculación']).dt.days // 30

    return df_resultado

# Ejecución
def main():

    # Cargamos los datos
    df_resultado = pd.read_json("../adquisicion/datosCrudos.json", orient="index")

    # Ejecutamos la limpieza
    df_resultado = limpieza(df_resultado)

    # Guardamos los datos procesados en formato parquet
    # (Puede ser necesario instalar las dependencias pyarrow y fastparquet)
    df_resultado.to_parquet("datosProcesados.parquet", index=0)


if __name__ == "__main__":
    main()
