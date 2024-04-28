# Librerias
from scipy.stats import loguniform
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm
from funciones import *
import numpy as np
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

def apply_pca(X, n_components=20):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def calculate_vif(X_pca):
    vif_data = pd.DataFrame()
    vif_data['VIF'] = [variance_inflation_factor(X_pca, i) for i in range(X_pca.shape[1])]
    vif_data['feature'] = range(X_pca.shape[1])
    return vif_data


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
    ID_DATOS = '1N9ucOBFVTi1A-LOU1UyqmU_-pUwR40SY'
    data = cargar_parquet_drive(ID_DATOS)
    X = data.drop(columns=["Precio(€)", "Accidentado", "Adicional", "Primera matriculación"])
    y = data['Precio(€)']

    # Transformación y PCA
    X_transformed, scaler = transformData(X)
    X_pca, pca = apply_pca(X_transformed)
    vif_data = calculate_vif(pd.DataFrame(X_pca))
    print(vif_data)

    X_train, X_test, y_train, y_test = splitTrainTest(X_pca, y)

    reg = LinearRegression()
    ajusteHiperparametros(X_train, y_train, X_test, y_test, reg, mlflow=True)

    # Uso de los mejores parámetros (se debe extraer de GridSearch si necesario)
    reg = LinearRegression(fit_intercept=True, copy_X= False)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("Precisión del modelo (R^2):", reg.score(X_test, y_test))
    analisisResultado(y_test, y_pred)
    plotModelo(y_test, y_pred, "Linear Regression with PCA")


if __name__ == "__main__":
    main()
