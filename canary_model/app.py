# FastAPI conversion of flask_canary.py
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from typing import Dict, List, Any

app = FastAPI(title="Canary Model API - Naive Bayes")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model training
iris = datasets.load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=0
)
loaded_model = GaussianNB()
loaded_model.fit(X_train, Y_train)

# Calculate and log model accuracy
train_accuracy = loaded_model.score(X_train, Y_train)
test_accuracy = loaded_model.score(X_test, Y_test)
logger.info(
    f"Canary model trained with Gaussian Naive Bayes. Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}"
)


class IrisData(BaseModel):
    s_l: float
    s_w: float
    p_l: float
    p_w: float


def predict_data(data_dict: Dict[str, float]) -> Dict[str, Any]:
    data = pd.DataFrame(data_dict, index=[0])

    # Get probability scores
    prediction_proba = loaded_model.predict_proba(data).tolist()[0]

    # Get the predicted class
    prediction_class = loaded_model.predict(data)[0]

    # Map class index to iris species name
    class_names = {0: "setosa", 1: "versicolor", 2: "virginica"}

    predicted_species = class_names.get(prediction_class, "unknown")

    return {
        "probability_scores": prediction_proba,
        "predicted_class": int(prediction_class),
        "predicted_species": predicted_species,
    }


@app.post("/predict")
async def predict(data: IrisData):
    try:
        data_dict = data.dict()
        logger.info(f"Received prediction request with data: {data_dict}")

        prediction_results = predict_data(data_dict)

        response = {
            "model_type": "GaussianNB",
            "predictions": prediction_results,
            "input_data": data_dict,
        }

        logger.info(
            f"Prediction result: Species={prediction_results['predicted_species']} with probability={max(prediction_results['probability_scores']):.4f}"
        )
        return response
    except Exception as ex:
        logger.error(f"Error during prediction: {str(ex)}")
        raise HTTPException(status_code=400, detail=str(ex))


@app.get("/health")
async def health_check():
    """Endpoint for health check monitoring"""
    return {"status": "healthy", "model": "GaussianNB"}


@app.get("/metadata")
async def model_metadata():
    """Return model metadata for model governance"""
    return {
        "model_type": "GaussianNB",
        "feature_names": ["sepal length", "sepal width", "petal length", "petal width"],
        "target_classes": ["setosa", "versicolor", "virginica"],
        "metrics": {
            "training_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
        },
        "version": "1.0.0",
    }


if __name__ == "__main__":
    # This block is only used when running the file directly, not when imported
    # Run with: uvicorn app:app --host 0.0.0.0 --port 5001 --reload
    logger.info("Starting Canary Model API (Naive Bayes) on port 5001")
