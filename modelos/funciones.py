# Librerias
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO


def cargar_parquet_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    file = requests.get(url)
    bytes_io = BytesIO(file.content)
    return pd.read_parquet(bytes_io)

# Función que separa en train y test empleando la libreria correspondiente de Scikit-Learn
def splitTrainTest(X, y, testSize=0.3, randomState=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)
    return X_train, X_test, y_train, y_test


def analisisResultado(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    print("\nMAE: ", mae)

# Método que aplica diversas técnicas de preprocesado para transformar los datos
def transformData(datos):
    # Aplicamos OneHotEncoder a las variables categóricas (las de tipo string)
    toEncode = ["Marca", "Modelo", "Carburante", "Transmisión", "Tracción"]
    datosCat = datos[toEncode]
    encoder = OneHotEncoder(drop='if_binary', sparse_output=False)
    datosEncodes = encoder.fit_transform(datosCat)
    features = encoder.get_feature_names_out()
    datosCat = pd.DataFrame(datosEncodes, columns=features)
    dataTrans = pd.concat([datos.drop(columns=toEncode), datosCat], axis=1)

    # También estandarizamos estos valores numéricos
    estandarizar = ["Kilometraje(Km)", "Potencia(Cv)", "Cilindrada(Cc)", "Edad(Meses)"]
    scaler = StandardScaler()
    dataTrans[estandarizar] = scaler.fit_transform(dataTrans[estandarizar])
    return dataTrans, scaler


def plotModelo(y_test, y_pred, nombre_modelo):
    # Diagrama de dispersión de predicciones vs. valores reales
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.title(f'Diagrama de dispersión de predicciones vs. valores reales (Modelo de {nombre_modelo})')
    plt.grid(True)
    plt.show()

def MLFlow(X_train, y_train, X_test, y_test, search, title):
    # Configurar MLFlow para usar SQLite como backend
    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'

    # Creamos (o elegimos nuestro experimento)
    mlflow.set_experiment(title)

    # Iniciar una nueva ejecución de MLFlow
    with mlflow.start_run():
        # Realizar la búsqueda de hiperparámetros
        search.fit(X_train, y_train)

        # Registrar el modelo en MLFlow
        mlflow.sklearn.log_model(search.best_estimator_, "model")

        # Registrar los hiperparámetros y la puntuación del mejor modelo
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("MAE", mean_absolute_error(y_test, search.predict(X_test)))