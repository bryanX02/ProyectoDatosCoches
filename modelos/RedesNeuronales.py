#################################################
# Modelo de regresión mediante Redes Neuronales #
#################################################

# En este programa se modelizará la regresión predictora del precio de los coches empleando un perceptron multicapa.
# Para ello emplearemos un las conocidas librerías de aprendizajé automático [Scikit-Learn]

# Librerias
from scipy.stats import loguniform
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from tqdm import tqdm
from funciones import *
import numpy as np

def ajusteHiperparametros(X_train, y_train, X_test, y_test, model, mlflow = False):

    # Definimos los hiperparámetros que se combinaran. Son 108 combinaciones
    param_grid = {
        'hidden_layer_sizes': [(50, 25), (100, 50), (150, 100)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'early_stopping': [True, False]
    }

    # Definición de hiperparámetros a probar
    param_range = {
        'hidden_layer_sizes': [(i, j) for i in range(50, 151, 1) for j in range(25, 101, 1)],
        'activation': ['relu', 'tanh', 'logistic'],
        'alpha': loguniform(1e-4, 1e-2),
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'early_stopping': [True, False]
    }

    # Definimos la validación cruzada
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Búsqueda en rejilla
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
    # Búsqueda aleatoria (limitamos la busqueda a 100
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_range, n_iter=100, cv=cv)

    if not mlflow:

        grid_search.fit(X_train, y_train.ravel())

        # Resultados de la búsqueda en rejilla
        print("Mejores hiperparámetros encontrados mediante búsqueda en rejilla:")
        print(grid_search.best_params_)
        print("MSE en entrenamiento:", -grid_search.best_score_)

        random_search.fit(X_train, y_train.ravel())

        # Resultados de la búsqueda aleatoria
        print("\nMejores hiperparámetros encontrados mediante búsqueda aleatoria:")
        print(random_search.best_params_)
        print("MSE en entrenamiento:", -random_search.best_score_)

    else:

        # Realizamos la busqueda y la guardamos en MLFlow
        MLFlow(X_train, y_train.ravel(), X_test, y_test.ravel(), grid_search, "Grid Search")
        print("Grid listo")
        MLFlow(X_train, y_train.ravel(), X_test, y_test.ravel(), grid_search, "Random Search")


def main():

    # Id de los datos procesados en formato parquet, almacenados en drive
    ID_DATOS = '1N9ucOBFVTi1A-LOU1UyqmU_-pUwR40SY'

    # Se cargan los datos
    data = cargar_parquet_drive(ID_DATOS)

    # Separación entre variables de entrada y objetivo
    X = data.drop(columns=["Precio(€)", "Accidentado", "Adicional", "Primera matriculación"])
    y = data[['Precio(€)']]
    # En la memoria explicamos el porqué no usamos las variables "Accidentado", "Adicional" y "Primera matriculación"

    # Ahora aplicamos las transformaciones necesarias
    X, scaler = transformData(X)
    X = X.to_numpy()
    y = y.to_numpy()

    # Con los datos ya listos, separamos entre train y test
    X_train, X_test, y_train, y_test = splitTrainTest(X, y)

    # Generamos el modelo y estudiamos que hiperparámetros son los óptimos, a la vez que guardamos el proceso en MLFlow
    reg = MLPRegressor()
    ajusteHiperparametros(X_train, y_train, X_test, y_test, reg, mlflow=True)

    # Y ahora ya ajustamos el modelo con los hiperparámetros obtenidos
    reg = MLPRegressor(hidden_layer_sizes=(150, 100), alpha=0.001, activation='relu', random_state=42, max_iter=1000)

    # Inicializamos la barra de progreso
    pbar = tqdm(total=reg.max_iter, desc="Entrenando modelo", position=0, leave=True)

    # Entrenamos el modelo en mini-batches
    batch_size = 1
    for epoch in range(int(np.ceil(reg.max_iter / batch_size))):
        # Obtener el índice de inicio y final del lote actual
        start_idx = epoch * batch_size
        end_idx = min((epoch + 1) * batch_size, reg.max_iter)

        # Entrenar el modelo en el lote actual
        reg.partial_fit(X_train, y_train.ravel())

        # Actualizar la barra de progreso
        pbar.update(end_idx - start_idx)

    # Cerrar la barra de progreso
    pbar.close()
    # Evaluamos el modelo en el conjunto de prueba
    y_pred = reg.predict(X_test)

    precisionTest = reg.score(X_test, y_test.ravel())
    print("Precisión del modelo (R^2):", precisionTest)
    analisisResultado(y_test, y_pred)
    plotModelo(y_test, y_pred, "Redes neuronales")


if __name__ == "__main__":
    main()
