import os
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


# Configurar MLFlow para usar SQLite como backend
os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'


# Cargar los datos
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo
model = RandomForestClassifier()

# Definir los hiperparámetros a ajustar
param_dist = {"max_depth": [3, None],
              "n_estimators": range(10, 200),
              "max_features": range(1, 4),
              "min_samples_split": range(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# Definir la validación cruzada
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Definir la búsqueda aleatoria
random_search = RandomizedSearchCV(model, param_distributions=param_dist, cv=cv, n_iter=10)

# Creamos (o elegimos nuestro experimento)
mlflow.set_experiment("Random search")

# Iniciar una nueva ejecución de MLFlow
with mlflow.start_run():
    # Realizar la búsqueda de hiperparámetros
    random_search.fit(X_train, y_train)

    # Registrar el modelo en MLFlow
    mlflow.sklearn.log_model(random_search.best_estimator_, "model")

    # Registrar los hiperparámetros y la puntuación del mejor modelo
    mlflow.log_params(random_search.best_params_)
    mlflow.log_metric("accuracy", accuracy_score(y_test, random_search.predict(X_test)))