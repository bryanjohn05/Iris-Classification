import subprocess
import sys
import os
import time
import requests
from pathlib import Path

def check_python_packages():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'joblib', 'scikit-learn', 
        'numpy', 'pandas', 'python-multipart'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstalling missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
    
    return True

def check_model_files():
    """Check if model files exist"""
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir.mkdir()
        print("📁 Created models directory")
    
    required_files = [
        "models/scaler.joblib",
        "models/RandomForest_model.joblib", 
        "models/SVM_model.joblib",
        "models/KNN_model.joblib",
        "models/DecisionTree_model.joblib",
        "models/LogisticRegression_model.joblib"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Missing model files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 To generate model files, run your training script first.")
        print("   The service will still start but predictions will be simulated.")
        return False
    else:
        print("✅ All model files found!")
        return True

def start_service():
    """Start the FastAPI service"""
    try:
        print("🚀 Starting Python backend service...")
        
        # Create the service file content
        service_code = '''
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List, Dict
import os
from pathlib import Path

app = FastAPI(title="Iris Classification API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to load models
models = {}
scaler = None

try:
    import joblib
    if Path("models/scaler.joblib").exists():
        scaler = joblib.load("models/scaler.joblib")
        print("✅ Scaler loaded")
    
    model_files = {
        "Random Forest": "models/RandomForest_model.joblib",
        "SVM": "models/SVM_model.joblib", 
        "KNN": "models/KNN_model.joblib",
        "Decision Tree": "models/DecisionTree_model.joblib",
        "Logistic Regression": "models/LogisticRegression_model.joblib",
    }
    
    for name, file_path in model_files.items():
        if Path(file_path).exists():
            models[name] = joblib.load(file_path)
            print(f"✅ {name} model loaded")
        else:
            print(f"⚠️  {name} model file not found: {file_path}")
            
except ImportError as e:
    print(f"❌ Error importing joblib: {e}")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# Species mapping
species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

class PredictionRequest(BaseModel):
    model: str
    features: List[float]

class PredictionAllRequest(BaseModel):
    features: List[float]

def simulate_prediction(features, model_name):
    """Fallback simulation when models aren't available"""
    sepal_length, sepal_width, petal_length, petal_width = features
    
    # Simple rule-based classification
    if petal_length <= 2.5:
        prediction = "Iris-setosa"
        confidence = 0.95
    elif petal_width <= 1.7:
        prediction = "Iris-versicolor" 
        confidence = 0.88
    else:
        prediction = "Iris-virginica"
        confidence = 0.92
    
    # Create probabilities
    probabilities = {
        "Iris-setosa": 0.05,
        "Iris-versicolor": 0.05, 
        "Iris-virginica": 0.05
    }
    probabilities[prediction] = confidence
    
    # Normalize
    remaining = 1 - confidence
    other_species = [s for s in probabilities.keys() if s != prediction]
    for species in other_species:
        probabilities[species] = remaining / len(other_species)
    
    return {
        "model": model_name,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities
    }

@app.get("/")
async def root():
    return {
        "message": "Iris Classification API is running!",
        "models_loaded": len(models),
        "scaler_loaded": scaler is not None
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "models": list(models.keys())}

@app.post("/predict")
async def predict(request: PredictionRequest):
    if len(request.features) != 4:
        raise HTTPException(status_code=400, detail="Expected 4 features")
    
    # Use real model if available
    if request.model in models and scaler is not None:
        try:
            input_data = np.array([request.features])
            input_scaled = scaler.transform(input_data)
            
            model = models[request.model]
            prediction = model.predict(input_scaled)[0]
            prediction_name = species_map[prediction]
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[0]
                probabilities = {species_map[i]: float(prob) for i, prob in enumerate(proba)}
                confidence = float(max(proba))
            else:
                confidence = 0.85
                probabilities = {species_map[i]: 0.33 for i in range(3)}
                probabilities[prediction_name] = confidence
            
            return {
                "model": request.model,
                "prediction": prediction_name,
                "confidence": confidence,
                "probabilities": probabilities
            }
        except Exception as e:
            print(f"Model prediction error: {e}")
            return simulate_prediction(request.features, request.model)
    else:
        # Fallback to simulation
        return simulate_prediction(request.features, request.model)

@app.post("/predict-all")
async def predict_all(request: PredictionAllRequest):
    if len(request.features) != 4:
        raise HTTPException(status_code=400, detail="Expected 4 features")
    
    model_names = ["Random Forest", "SVM", "KNN", "Decision Tree", "Logistic Regression"]
    predictions = []
    
    for model_name in model_names:
        if model_name in models and scaler is not None:
            try:
                input_data = np.array([request.features])
                input_scaled = scaler.transform(input_data)
                
                model = models[model_name]
                prediction = model.predict(input_scaled)[0]
                prediction_name = species_map[prediction]
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_scaled)[0]
                    probabilities = {species_map[i]: float(prob) for i, prob in enumerate(proba)}
                    confidence = float(max(proba))
                else:
                    confidence = 0.85
                    probabilities = {species_map[i]: 0.33 for i in range(3)}
                    probabilities[prediction_name] = confidence
                
                predictions.append({
                    "model": model_name,
                    "prediction": prediction_name,
                    "confidence": confidence,
                    "probabilities": probabilities
                })
            except Exception as e:
                print(f"Model {model_name} prediction error: {e}")
                predictions.append(simulate_prediction(request.features, model_name))
        else:
            predictions.append(simulate_prediction(request.features, model_name))
    
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        # Write service file
        with open("temp_service.py", "w") as f:
            f.write(service_code)
        
        # Start the service
        process = subprocess.Popen([
            sys.executable, "temp_service.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for service to start
        time.sleep(3)
        
        # Test if service is running
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Service started successfully!")
                print("🌐 API available at: http://localhost:8000")
                print("📚 API docs at: http://localhost:8000/docs")
                print("🔍 Health check: http://localhost:8000/health")
                print("\n🎯 You can now use the frontend to make predictions!")
                print("   The service will use real models if available, or simulate predictions otherwise.")
                
                # Keep the service running
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\n🛑 Stopping service...")
                    process.terminate()
                    
            else:
                print(f"❌ Service health check failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to service: {e}")
            print("Service may still be starting up...")
            
    except Exception as e:
        print(f"❌ Failed to start service: {e}")
    finally:
        # Clean up temp file
        if os.path.exists("temp_service.py"):
            os.remove("temp_service.py")

def main():
    print("🌸 Iris Classification Service Starter")
    print("=" * 50)
    
    # Check dependencies
    if not check_python_packages():
        print("❌ Failed to install required packages")
        return
    
    # Check model files
    models_available = check_model_files()
    
    if models_available:
        print("🎉 Ready to start service with real models!")
    else:
        print("⚠️  Starting service with simulated predictions")
    
    print("\n" + "=" * 50)
    start_service()

if __name__ == "__main__":
    main()
