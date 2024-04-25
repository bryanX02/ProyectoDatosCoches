#################################################
# Modelo de regresión mediante Redes Neuronales #
#################################################

# En este programa se modelizará la regresión predictora del precio de los coches empleando un perceptron multicapa.
# Para ello emplearemos un las conocidas librerías de aprendizajé automático [Scikit-Learn]

# Librerias

import pandas as pd
from scipy.stats import loguniform
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from funciones import *
import matplotlib.pyplot as plt
import numpy as np
import os
import mlflow
import mlflow.sklearn


def ajusteHiperparametros(X_train, y_train, model):
    # Definimos los hiperparámetros que se combinaran. Son 18 (ya que en este modelo se requieren muchos recursos)
    param_grid = {
        'hidden_layer_sizes': [(50, 25), (100, 50), (150, 100)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01]
    }

    # Definición de hiperparámetros a probar
    param_range = {
        'hidden_layer_sizes': [(i, j) for i in range(50, 151, 50) for j in range(25, 101, 25)],
        'activation': ['relu', 'tanh', 'logistic'],
        'alpha': loguniform(1e-4, 1e-2),
    }

    # Definimos la validación cruzada
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Búsqueda en rejilla
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
    grid_search.fit(X_train, y_train)

    # Resultados de la búsqueda en rejilla
    print("Mejores hiperparámetros encontrados mediante búsqueda en rejilla:")
    print(grid_search.best_params_)
    print("MSE en entrenamiento:", -grid_search.best_score_)

    # Búsqueda aleatoria
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_range, n_iter=27, cv=cv)
    random_search.fit(X_train, y_train)

    # Resultados de la búsqueda aleatoria
    print("\nMejores hiperparámetros encontrados mediante búsqueda aleatoria:")
    print(random_search.best_params_)
    print("MSE en entrenamiento:", -random_search.best_score_)




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


def plotRedesNeuronales(y_test, y_pred):
    # Diagrama de dispersión de predicciones vs. valores reales
    print("Primeras predicciones:", y_test[:10])
    print("Primeras predicciones:", y_pred[:10])
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.title('Diagrama de dispersión de predicciones vs. valores reales (GridSearch)')
    plt.grid(True)
    plt.show()


def main():
    # Se cargan los datos
    data = pd.read_parquet("../limpieza/datosProcesados.parquet")

    # Separación entre variables de entrada y objetivo
    X = data.drop(columns=["Precio(€)", "Accidentado", "Adicional", "Primera matriculación"])
    y = data["Precio(€)"]
    # En la memoria explicamos el porque no usamos las variables "Accidentado", "Adicional" y "Primera matriculación"

    # Ahora aplicamos las transformaciones necesarias
    X, scaler = transformData(X)

    # Con los datos ya listos, separamos entre train y test
    X_train, X_test, y_train, y_test = splitTrainTest(X, y)

    # Generamos el modelo y estudiamos que hiperparámetros son los óptimos
    reg = MLPRegressor(max_iter=1000)
    # ajusteHiperparametros(X_train, y_train, reg)

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
        reg.partial_fit(X_train.values, y_train.values.ravel())

        # Actualizar la barra de progreso
        pbar.update(end_idx - start_idx)

    # Cerrar la barra de progreso
    pbar.close()
    # Evaluamos el modelo en el conjunto de prueba
    y_pred = reg.predict(X_test)

    precisionTest = reg.score(X_test.values, y_test.values.ravel())
    print("Precisión del modelo (R^2):", precisionTest)
    analisisResultado(y_test, y_pred)

    # COMENTAR LO SIGUIENTE SI NO SE QUIERE GENERAR EL MLFLOW:
    '''
    # Por último registramos el experimento con MLFlow
    # Definimos los hiperparámetros que se combinaran. Son 18 (ya que en este modelo se requieren muchos recursos)
    param_grid = {
        'hidden_layer_sizes': [(50, 25), (100, 50), (150, 100)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01]
    }

    # Definición de hiperparámetros a probar
    param_range = {
        'hidden_layer_sizes': [(i, j) for i in range(50, 151, 50) for j in range(25, 101, 25)],
        'activation': ['relu', 'tanh', 'logistic'],
        'alpha': loguniform(1e-4, 1e-2),
    }

    model = MLPRegressor(max_iter=1000)
    random_search = GridSearchCV(model, param_grid=param_grid, cv=3)
    MLFlow(X_train, y_train, X_test, y_test, random_search, "Grid Search")
    random_search = RandomizedSearchCV(model, param_distributions=param_range, cv=3, n_iter=10)
    MLFlow(X_train, y_train, X_test, y_test, random_search, "Random Search")'''
    y_pred = y_pred.reshape(-1, 1)
    plotRedesNeuronales(y_test, y_pred)


if __name__ == "__main__":
    main()
