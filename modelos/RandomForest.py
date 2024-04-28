

from scipy.stats import loguniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from tqdm import tqdm
from funciones import *
import numpy as np
from scipy.stats import randint

def ajusteHiperparametros(X_train, y_train, X_test, y_test, model, mlflow = False):

    # Definir el rango de valores de n_estimators y max_leaf_nodes a probar para la búsqueda en cuadrícula
    param_grid_grid_search = {
        'n_estimators': [50, 100, 150, 200],
        'max_leaf_nodes': [None, 10, 20, 30]
    }

    # Definir el rango de valores de n_estimators y max_leaf_nodes a probar para la búsqueda aleatoria
    param_dist_random_search = {
        'n_estimators': randint(50, 200),
        'max_leaf_nodes': [None, 10, 20, 30]
    }

    # Definimos la validación cruzada
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Búsqueda en rejilla
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid_grid_search, cv=cv)
    # Búsqueda aleatoria (limitamos la busqueda a 100
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist_random_search, n_iter=100, cv=cv)

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
    ajusteHiperparametros(X_train, y_train, X_test, y_test, reg, mlflow=True)

    # Y ahora ya ajustamos el modelo con los hiperparámetros obtenidos
    reg = RandomForestRegressor(random_state=RANDOM_STATE) # Añadir los mejores hiperparámetros
    reg.fit(X_train, y_train.ravel())

    # Evaluamos el modelo en el conjunto de prueba
    y_pred = reg.predict(X_test)

    precisionTest = reg.score(X_test, y_test.ravel())
    print("Precisión del modelo (R^2):", precisionTest)
    analisisResultado(y_test, y_pred)
    plotModelo(y_test, y_pred, "RandomForest")


if __name__ == "__main__":
    main()


