# Librerias
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

# Función que separa en train y test empleando la libreria correspondiente de Scikit-Learn
def splitTrainTest(X, y, testSize=0.3, randomState=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)
    return X_train, X_test, y_train, y_test


def analisisResultado(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(y_test, y_pred)
    print("\nRMSE: ", rmse, "\n\nMAE: ", mae)

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
