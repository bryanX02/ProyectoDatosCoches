# Librerias
from scipy.stats import loguniform
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm
from funciones import *
import numpy as np

def ajusteHiperparametros(X_train, y_train, X_test, y_test, model, mlflow = False):

    # Definimos los hiperparámetros que se combinaran. Son 108 combinaciones
    param_grid = {
        'fit_intercept': [True, False],
        'copy_X': [True, False]
    }

    # Definimos la validación cruzada
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Búsqueda en rejilla
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
    if not mlflow:

        grid_search.fit(X_train, y_train.ravel())

        # Resultados de la búsqueda en rejilla
        print("Mejores hiperparámetros encontrados mediante búsqueda en rejilla:")
        print(grid_search.best_params_)
        print("MSE en entrenamiento:", -grid_search.best_score_)

    else:

        # Realizamos la busqueda y la guardamos en MLFlow
        MLFlow(X_train, y_train.ravel(), X_test, y_test.ravel(), grid_search, "Grid Search")
        print("Grid listo")

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
    reg = LinearRegression()
    ajusteHiperparametros(X_train, y_train, X_test, y_test, reg, mlflow=True)

    # Y ahora ya ajustamos el modelo con los hiperparámetros obtenidos
    reg = LinearRegression(fit_intercept=False, copy_X=True) # Añadir mejores hiperparámetros
    reg.fit(X_train, y_train)
    # Evaluamos el modelo en el conjunto de prueba
    y_pred = reg.predict(X_test)

    precisionTest = reg.score(X_test, y_test.ravel())
    print("Precisión del modelo (R^2):", precisionTest)
    analisisResultado(y_test, y_pred)
    plotModelo(y_test, y_pred, "Linear Regression")


if __name__ == "__main__":
    main()
