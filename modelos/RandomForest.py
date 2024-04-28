

from scipy.stats import loguniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from tqdm import tqdm
from funciones import *
import numpy as np
from scipy.stats import randint
from sklearn.metrics import mean_absolute_error
import random

def ajusteHiperparametros(X_train, y_train, X_test, y_test, model, mlflow = False):

    # Definir el rango de valores de n_estimators y max_leaf_nodes a probar para la búsqueda en cuadrícula
    param_grid_grid_search = {
    'n_estimators': [50, 100, 150],
    'max_leaf_nodes': [10, 50, 100, None],
    'max_features': [0.2, 0.5, 0.7, 1], #se pude poner en porcetaje(0-1)
    'min_samples_split': [2, 10, 50, 100],
    'max_depth': [10, 50, 100, None],
    'max_samples': [0.2, 0.5, 0.9, None]

    }

    # Definir el rango de valores de n_estimators y max_leaf_nodes a probar para la búsqueda aleatoria
    param_dist_random_search = {
    'n_estimators': range(50, 151),
    'max_leaf_nodes': [10, 50, 100, None],
    'max_features': [0.2, 0.5, 0.7, 1],
    'min_samples_split': range(2, 101),
    'max_depth': [10, 50, 100, None],
    'max_samples': [0.2, 0.5, 0.9, None]
    }
    # Crear el modelo de Random Forest
    # Definimos la validación cruzada
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Búsqueda en rejilla
    grid_search = GridSearchCV(model, param_grid_grid_search, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    # Búsqueda aleatoria (limitamos la busqueda a 100
    # Inicializar RandomizedSearchCV con el modelo, el espacio de búsqueda y la métrica de evaluación
    random_search = RandomizedSearchCV(model, param_dist_random_search, cv=cv, scoring='neg_mean_absolute_error', n_iter=100, n_jobs=-1)

    if not mlflow:

        grid_search.fit(X_train, y_train.ravel())

        # Resultados de la búsqueda en rejilla
        print("Mejores hiperparámetros encontrados mediante búsqueda en rejilla:")
        print(grid_search.best_params_)
        print("MAE en entrenamiento:", -grid_search.best_score_)

        random_search.fit(X_train, y_train.ravel())

        # Resultados de la búsqueda aleatoria
        print("\nMejores hiperparámetros encontrados mediante búsqueda aleatoria:")
        print(random_search.best_params_)
        print("MAE en entrenamiento:", -random_search.best_score_)

    else:

        # Realizamos la busqueda y la guardamos en MLFlow
        MLFlow(X_train, y_train.ravel(), X_test, y_test.ravel(), grid_search, "Grid Search")
        print("Grid listo")
        MLFlow(X_train, y_train.ravel(), X_test, y_test.ravel(), random_search, "Random Search")
    # Obtener los mejores hiperparámetros de la búsqueda en cuadrícula
    best_params_grid_search = grid_search.best_params_

    # Extraer los valores de los hiperparámetros
    n_estimators_grid_search = best_params_grid_search['n_estimators']
    max_leaf_nodes_grid_search = best_params_grid_search['max_leaf_nodes']
    max_features_grid_search = best_params_grid_search['max_features']
    min_samples_split_grid_search = best_params_grid_search['min_samples_split']
    max_depth_grid_search = best_params_grid_search['max_depth']
    max_samples_grid_search = best_params_grid_search['max_samples']

    # Crear el modelo de Random Forest con los mejores hiperparámetros de la búsqueda en cuadrícula
    rand_forest_best_grid_search = RandomForestRegressor(n_estimators=n_estimators_grid_search, max_leaf_nodes=max_leaf_nodes_grid_search, max_features=max_features_grid_search, min_samples_split=min_samples_split_grid_search, max_depth=max_depth_grid_search, max_samples=max_samples_grid_search,random_state=42)
    # Obtener los mejores hiperparámetros de la búsqueda aleatoria
    best_params_random_search = random_search.best_params_

    # Extraer los valores de los hiperparámetros
    n_estimators_random_search = best_params_random_search['n_estimators']
    max_leaf_nodes_random_search = best_params_random_search['max_leaf_nodes']
    max_features_random_search = best_params_random_search['max_features']
    min_samples_split_random_search = best_params_random_search['min_samples_split']
    max_depth_random_search = best_params_random_search['max_depth']
    max_samples_random_search = best_params_random_search['max_samples']

    # Crear el modelo de Random Forest con los mejores hiperparámetros de la búsqueda aleatoria
    rand_forest_best_random_search = RandomForestRegressor(n_estimators=n_estimators_random_search, max_leaf_nodes=max_leaf_nodes_random_search, max_features=max_features_random_search, min_samples_split=min_samples_split_random_search, max_depth=max_depth_random_search, max_samples=max_samples_random_search, random_state=42)

    return rand_forest_best_grid_search, rand_forest_best_random_search

def main():

    RANDOM_STATE = 42

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
    reg = RandomForestRegressor(random_state=RANDOM_STATE)
    rand_forest_best_grid_search, rand_forest_best_random_search = ajusteHiperparametros(X_train, y_train, X_test, y_test, reg, mlflow=True)
    # Y ahora ya ajustamos el modelo con los hiperparámetros obtenidos
    rand_forest_best_grid_search.fit(X_train,y_train.ravel())
    rand_forest_best_random_search.fit(X_train,y_train.ravel())

    # Evaluamos el modelo en el conjunto de prueba
    predicciones_randF_G = rand_forest_best_grid_search.predict(X_test)
    predicciones_randF_S = rand_forest_best_random_search.predict(X_test)

    mae_RF_G = mean_absolute_error(y_test, predicciones_randF_G)
    mae_RF_S = mean_absolute_error(y_test, predicciones_randF_S)

    if(mae_RF_G <= mae_RF_S):
        y_pred = mae_RF_G
        print('MAE del modelo GridSearch: ', mae_RF_G, ' < ', 'MAE del modelo RandomSearch: ', mae_RF_S)
        reg = rand_forest_best_grid_search
    else:
        y_pred = mae_RF_S
        print('MAE del modelo RandomSearch: ', mae_RF_S, ' < ', 'MAE del modelo GridSearch: ', mae_RF_G)
        reg = rand_forest_best_random_search

    precisionTest = reg.score(X_test, y_test.ravel())
    print("Precisión del modelo (R^2):", precisionTest)
    analisisResultado(y_test, y_pred)
    plotModelo(y_test, y_pred, "RandomForest")


if __name__ == "__main__":
    main()


