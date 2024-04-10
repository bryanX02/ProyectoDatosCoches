# En el encabezado de librerías
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pandas as pd
import numpy as np

# Función que separa en train y test empleando la libreria correspondiente de Scikit-Learn
def splitTrainTest(X, y, testSize=0.3, randomState=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)
    return X_train, X_test, y_train, y_test

# Función que
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
    dataTrans[estandarizar] = StandardScaler().fit_transform(dataTrans[estandarizar])

    return dataTrans

# En la función main()
def main():
    # Se cargan los datos
    data = pd.read_parquet("../limpieza/datosProcesados.parquet")

    # Separación entre variables de entrada y objetivo
    X = data.drop(columns=["Precio(€)", "Accidentado", "Adicional", "Primera matriculación"])
    y = data["Precio(€)"]

    # Ahora aplicamos las transformaciones necesarias
    X = transformData(X)

    # Con los datos ya listos, separamos entre train y test
    X_train, X_test, y_train, y_test = splitTrainTest(X, y)

    # Creamos el modelo MLPRegressor con parámetros modificados
    reg = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', random_state=42, max_iter=1000)

    # Inicializamos la barra de progreso
    pbar = tqdm(total=reg.max_iter, desc="Entrenando modelo", position=0, leave=True)

    # Entrenamos el modelo en mini-batches
    batch_size = 0.5
    for epoch in range(int(np.ceil(reg.max_iter / batch_size))):
        # Obtener el índice de inicio y final del lote actual
        start_idx = epoch * batch_size
        end_idx = min((epoch + 1) * batch_size, reg.max_iter)

        # Entrenar el modelo en el lote actual
        reg.partial_fit(X_train.values, y_train.values.ravel())

        # Actualizar la barra de progreso
        pbar.update(end_idx - start_idx)

    # Cerrar la barra de progreso
    pbar.close()

    # Evaluamos el modelo en el conjunto de prueba
    y_pred = reg.predict(X_test.values)
    precisionTest = reg.score(X_test.values, y_test.values.ravel())
    mse = mean_squared_error(y_test, y_pred)

    print("Precisión del modelo (R^2):", precisionTest)
    print("Error cuadrático medio:", mse)


if __name__ == "__main__":
    main()
