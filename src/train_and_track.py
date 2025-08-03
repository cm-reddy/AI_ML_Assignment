import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Load preprocessed data
df = pd.read_csv("data/housing.csv")
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def train_and_log_model(model, model_name):
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Evaluate
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Log params and metrics
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        # Infer signature and input example
        signature = infer_signature(X_test, preds)
        input_example = X_test.head(2)

        # Log model with input example and signature
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

        print(f"✅ {model_name} | MSE: {mse:.3f} | R2 Score: {r2:.3f}")

# Train and log 2 models
train_and_log_model(LinearRegression(), "LinearRegression")
train_and_log_model(DecisionTreeRegressor(max_depth=5), "DecisionTree")
