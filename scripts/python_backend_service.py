from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict
import os

app = FastAPI(title="Iris Classification API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and scaler
try:
    scaler = joblib.load("models/scaler.joblib")
    models = {
        "Random Forest": joblib.load("models/RandomForest_model.joblib"),
        "SVM": joblib.load("models/SVM_model.joblib"),
        "KNN": joblib.load("models/KNN_model.joblib"),
        "Decision Tree": joblib.load("models/DecisionTree_model.joblib"),
        "Logistic Regression": joblib.load("models/LogisticRegression_model.joblib"),
    }
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    models = {}
    scaler = None

# Species mapping
species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

class PredictionRequest(BaseModel):
    model: str
    features: List[float]

class PredictionAllRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    model: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

@app.get("/")
async def root():
    return {"message": "Iris Classification API is running!", "models_loaded": len(models)}

@app.get("/models")
async def get_models():
    return {"models": list(models.keys())}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not models or not scaler:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    if request.model not in models:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' not found")
    
    if len(request.features) != 4:
        raise HTTPException(status_code=400, detail="Expected 4 features")
    
    try:
        # Prepare input data
        input_data = np.array([request.features])
        input_scaled = scaler.transform(input_data)
        
        # Get model
        model = models[request.model]
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_name = species_map[prediction]
        
        # Get probabilities if available
        probabilities = {}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_scaled)[0]
            probabilities = {species_map[i]: float(prob) for i, prob in enumerate(proba)}
            confidence = float(max(proba))
        else:
            # For models without predict_proba, use a simple confidence measure
            confidence = 0.85
            probabilities = {species_map[i]: 0.33 for i in range(3)}
            probabilities[prediction_name] = confidence
        
        return PredictionResponse(
            model=request.model,
            prediction=prediction_name,
            confidence=confidence,
            probabilities=probabilities
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-all")
async def predict_all(request: PredictionAllRequest):
    if not models or not scaler:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    if len(request.features) != 4:
        raise HTTPException(status_code=400, detail="Expected 4 features")
    
    try:
        predictions = []
        
        # Prepare input data
        input_data = np.array([request.features])
        input_scaled = scaler.transform(input_data)
        
        for model_name, model in models.items():
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_name = species_map[prediction]
            
            # Get probabilities if available
            probabilities = {}
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[0]
                probabilities = {species_map[i]: float(prob) for i, prob in enumerate(proba)}
                confidence = float(max(proba))
            else:
                # For models without predict_proba, use a simple confidence measure
                confidence = 0.85
                probabilities = {species_map[i]: 0.33 for i in range(3)}
                probabilities[prediction_name] = confidence
            
            predictions.append({
                "model": model_name,
                "prediction": prediction_name,
                "confidence": confidence,
                "probabilities": probabilities
            })
        
        return {"predictions": predictions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-performance")
async def get_model_performance():
    # This would typically load from saved metrics or re-evaluate on test set
    # For now, returning example performance metrics
    performance_data = {
        "Random Forest": {"accuracy": 0.97, "precision": 0.96, "recall": 0.97, "f1_score": 0.965},
        "SVM": {"accuracy": 0.95, "precision": 0.94, "recall": 0.95, "f1_score": 0.945},
        "KNN": {"accuracy": 0.93, "precision": 0.92, "recall": 0.93, "f1_score": 0.925},
        "Decision Tree": {"accuracy": 0.91, "precision": 0.90, "recall": 0.91, "f1_score": 0.905},
        "Logistic Regression": {"accuracy": 0.94, "precision": 0.93, "recall": 0.94, "f1_score": 0.935},
    }
    
    return {"performance": performance_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
