import os
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# # Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
mlflow.set_tracking_uri = MLFLOW_TRACKING_URI
client = MlflowClient()

# Prepare data
X, y = make_classification()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and calculate accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Start an MLflow run
with mlflow.start_run():

    # Log model
    mlflow.sklearn.log_model(
        model, artifact_path="tmp/mlartifacts", registered_model_name="model"
    )

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log params
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_params(
        {"max_depth": model.max_depth, "n_estimators": model.n_estimators}
    )
