import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os
import mlflow
import mlflow.sklearn


# Función para dividir los datos en entrenamiento y prueba
def split_train_test(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Transformación de datos con pipelines
def transform_data():
    categorical_features = ["Marca", "Modelo", "Carburante", "Transmisión", "Tracción"]
    numeric_features = ["Kilometraje(Km)", "Potencia(Cv)", "Cilindrada(Cc)", "Edad(Meses)"]

    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", OneHotEncoder(drop='if_binary'), categorical_features)
    ])

    return transformer


# Evaluación del modelo
def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print("\nResultados del modelo:")
    print("RMSE: ", rmse)
    print("MAE: ", mae)
    print("R²: ", r2)


# Registro y seguimiento con MLflow
def run_mlflow(X_train, y_train, X_test, y_test, model, title="Regresión Lineal Múltiple"):
    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'
    mlflow.set_experiment(title)

    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_params({"model_type": "LinearRegression"})
        mlflow.log_metrics({"MAE": mean_absolute_error(y_test, y_pred), "MSE": mean_squared_error(y_test, y_pred),
                            "R2": r2_score(y_test, y_pred)})


# Proceso principal
def main():
    data = pd.read_parquet("../limpieza/datosProcesados.parquet")
    X = data.drop(columns=["Precio(€)", "Accidentado", "Adicional", "Primera matriculación"])
    y = data["Precio(€)"]

    transformer = transform_data()
    X_transformed = transformer.fit_transform(X)

    X_train, X_test, y_train, y_test = split_train_test(X_transformed, y)
    reg = LinearRegression()
    print("Iniciando entrenamiento y evaluación del modelo...")
    run_mlflow(X_train, y_train, X_test, y_test, reg)

    y_pred = reg.predict(X_test)
    evaluate_model(y_test, y_pred)


if __name__ == "__main__":
    main()
