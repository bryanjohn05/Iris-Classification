import subprocess
import sys
import os

def install_requirements():
    """Install required Python packages"""
    requirements = [
        "fastapi",
        "uvicorn[standard]",
        "joblib",
        "scikit-learn",
        "numpy",
        "pandas",
        "python-multipart"
    ]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def create_models_directory():
    """Create models directory if it doesn't exist"""
    if not os.path.exists("models"):
        os.makedirs("models")
        print("📁 Created models directory")
    else:
        print("📁 Models directory already exists")

def check_model_files():
    """Check if model files exist"""
    model_files = [
        "models/scaler.joblib",
        "models/RandomForest_model.joblib",
        "models/SVM_model.joblib",
        "models/KNN_model.joblib",
        "models/DecisionTree_model.joblib",
        "models/LogisticRegression_model.joblib"
    ]
    
    missing_files = []
    for file in model_files:
        if os.path.exists(file):
            print(f"✅ Found {file}")
        else:
            print(f"❌ Missing {file}")
            missing_files.append(file)
    
    if missing_files:
        print("\n⚠️  Missing model files. Please ensure you have:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nRun your training script to generate these files.")
    else:
        print("\n🎉 All model files found!")

def main():
    print("🚀 Setting up Iris Classification Backend")
    print("=" * 50)
    
    install_requirements()
    print()
    create_models_directory()
    print()
    check_model_files()
    
    print("\n" + "=" * 50)
    print("Setup complete! To start the backend server:")
    print("1. Ensure your model files are in the 'models' directory")
    print("2. Run: python scripts/python_backend_service.py")
    print("3. The API will be available at http://localhost:8000")
    print("4. API documentation at http://localhost:8000/docs")

if __name__ == "__main__":
    main()
