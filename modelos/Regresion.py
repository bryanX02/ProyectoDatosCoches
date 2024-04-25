import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.compose import ColumnTransformer
import numpy as np
import os
import mlflow
import mlflow.sklearn

# Transformación de datos con pipelines
def transform_data():
    categorical_features = ["Marca", "Modelo", "Carburante", "Transmisión", "Tracción"]
    numeric_features = ["Kilometraje(Km)", "Potencia(Cv)", "Cilindrada(Cc)", "Edad(Meses)"]

    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", OneHotEncoder(drop='if_binary'), categorical_features)
    ])

    return transformer

# Evaluación del modelo con k-fold
def evaluate_model_kfold(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
               'MSE': make_scorer(mean_squared_error, greater_is_better=False),
               'R2': make_scorer(r2_score)}
    scores = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=False)
    print("\nResultados de validación cruzada:")
    print("RMSE: ", np.sqrt(-np.mean(scores['test_MSE'])))
    print("MAE: ", -np.mean(scores['test_MAE']))
    print("R²: ", np.mean(scores['test_R2']))

# Registro y seguimiento con MLflow
def run_mlflow(X, y, model, title="Regresión Lineal Múltiple"):
    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'
    mlflow.set_experiment(title)

    with mlflow.start_run():
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
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

    reg = LinearRegression()
    print("Validación cruzada del modelo...")
    evaluate_model_kfold(X_transformed, y, reg)
    run_mlflow(X_transformed, y, reg)

if __name__ == "__main__":
    main()
