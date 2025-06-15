import sys
import json
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_models_and_scaler():
    """Load all models and scaler"""
    models = {}
    scaler = None
    
    try:
        # Load scaler
        scaler_path = Path("models/scaler.joblib")
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        else:
            return None, None
        
        # Load models
        model_files = {
            "Random Forest": "models/RandomForest_model.joblib",
            "SVM": "models/SVM_model.joblib",
            "KNN": "models/KNN_model.joblib",
            "Decision Tree": "models/DecisionTree_model.joblib",
            "Logistic Regression": "models/LogisticRegression_model.joblib",
        }
        
        for name, filepath in model_files.items():
            model_path = Path(filepath)
            if model_path.exists():
                models[name] = joblib.load(model_path)
        
        return models, scaler
    except Exception as e:
        return None, None

def predict_single(model_name, features, models, scaler):
    """Make prediction with a single model"""
    if model_name not in models or scaler is None:
        raise ValueError(f"Model {model_name} not available")
    
    # Prepare input
    input_data = np.array([features])
    input_scaled = scaler.transform(input_data)
    
    # Get model and predict
    model = models[model_name]
    prediction_idx = model.predict(input_scaled)[0]
    
    # Species mapping
    species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    prediction_name = species_map[prediction_idx]
    
    # Get probabilities
    probabilities = {}
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_scaled)[0]
        probabilities = {species_map[i]: float(prob) for i, prob in enumerate(proba)}
        confidence = float(max(proba))
    else:
        confidence = 0.85
        probabilities = {species_map[i]: 0.33 for i in range(3)}
        probabilities[prediction_name] = confidence
    
    return {
        "model": model_name,
        "prediction": prediction_name,
        "confidence": confidence,
        "probabilities": probabilities
    }

def predict_all(features, models, scaler):
    """Make predictions with all models"""
    model_names = ["Random Forest", "SVM", "KNN", "Decision Tree", "Logistic Regression"]
    predictions = []
    
    for model_name in model_names:
        try:
            if model_name in models and scaler is not None:
                result = predict_single(model_name, features, models, scaler)
                predictions.append(result)
            else:
                # Skip unavailable models
                continue
        except Exception as e:
            # Skip failed predictions
            continue
    
    return {"predictions": predictions}

def test_environment():
    """Test if the environment is working"""
    try:
        models, scaler = load_models_and_scaler()
        if models and scaler:
            return {"status": "ready", "models_count": len(models)}
        else:
            return {"status": "models_not_available", "models_count": 0}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No command provided"}))
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "test":
            result = test_environment()
            print(json.dumps(result))
            
        elif command == "predict-single":
            if len(sys.argv) != 7:
                print(json.dumps({"error": "Invalid arguments for predict-single"}))
                sys.exit(1)
            
            # Load models and scaler
            models, scaler = load_models_and_scaler()
            
            if models is None or scaler is None:
                print(json.dumps({"error": "Models or scaler not available"}))
                sys.exit(1)
            
            model_name = sys.argv[2]
            features = [float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])]
            
            result = predict_single(model_name, features, models, scaler)
            print(json.dumps(result))
            
        elif command == "predict-all":
            if len(sys.argv) != 6:
                print(json.dumps({"error": "Invalid arguments for predict-all"}))
                sys.exit(1)
            
            # Load models and scaler
            models, scaler = load_models_and_scaler()
            
            if models is None or scaler is None:
                print(json.dumps({"error": "Models or scaler not available"}))
                sys.exit(1)
            
            features = [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])]
            
            result = predict_all(features, models, scaler)
            print(json.dumps(result))
            
        else:
            print(json.dumps({"error": f"Unknown command: {command}"}))
            sys.exit(1)
            
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
