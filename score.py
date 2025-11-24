import json
import numpy as np
import joblib
import os

def init():
    global model
    # Chemin du modèle monté par Azure ML
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model_path = os.path.join(model_dir, "isolation_forest_model.joblib")
    model = joblib.load(model_path)

def run(raw_data):
    try:
        # raw_data peut être une chaîne JSON ou déjà un dict
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data

        # Gérer deux formats possibles :
        # {"data": [...]} ou {"input_data": {"data": [...]}}
        if "input_data" in data:
            data = data["input_data"]

        features_list = data.get("data", data)

        # Conversion en numpy array
        features = np.array(features_list, dtype=float)

        # Prédictions IsolationForest
        predictions = model.predict(features)               # -1 = anomalie, 1 = normal
        scores = model.decision_function(features)          # score d'anomalie

        # Construire la réponse
        results = {
            "anomaly_predictions": predictions.tolist(),
            "anomaly_scores": scores.tolist(),
            "is_anomaly": [1 if p == -1 else 0 for p in predictions],
            "message": "Détection d'anomalies réussie",
            "anomalies_count": int((predictions == -1).sum())
        }

        # Azure ML sait sérialiser un dict en JSON
        return results

    except Exception as e:
        return {"error": str(e)}
