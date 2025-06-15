import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    Path("models").mkdir(exist_ok=True)
    print("📁 Created models directory")

def load_and_prepare_data():
    """Load and prepare the iris dataset"""
    try:
        # Try to load from URL first
        url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Iris-g2HfzvJAL5b6zEBQ7hZcKUkPSkzhVI.csv"
        df = pd.read_csv(url)
        print("✅ Loaded data from URL")
    except:
        # Fallback to sklearn dataset
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['Species'] = iris.target_names[iris.target]
        print("✅ Loaded data from sklearn")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Species distribution:\n{df['Species'].value_counts()}")
    
    return df

def preprocess_data(df):
    """Preprocess the data"""
    # Handle different column names
    if 'SepalLengthCm' in df.columns:
        # Original format
        feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        X = df[feature_cols]
    else:
        # sklearn format
        X = df.drop('Species', axis=1)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['Species'])
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Classes: {le.classes_}")
    
    return X, y, le

def train_and_save_models(X, y):
    """Train all models and save them"""
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, "models/scaler.joblib")
    print("✅ Saved scaler")
    
    # Define models and their parameters
    model_configs = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [None, 5, 10]
            }
        },
        "SVM": {
            "model": SVC(random_state=42, probability=True),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"]
            }
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5]
            }
        },
        "LogisticRegression": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {
                "C": [0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"]
            }
        }
    }
    
    results = {}
    
    # Train each model
    for name, config in model_configs.items():
        print(f"\n🔄 Training {name}...")
        
        # Grid search for best parameters
        grid_search = GridSearchCV(
            config["model"], 
            config["params"], 
            cv=3,  # Reduced for faster training
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        model_filename = f"models/{name}_model.joblib"
        joblib.dump(best_model, model_filename)
        
        results[name] = {
            "accuracy": accuracy,
            "best_params": grid_search.best_params_,
            "model_file": model_filename
        }
        
        print(f"✅ {name} trained successfully!")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Best params: {grid_search.best_params_}")
        print(f"   Saved to: {model_filename}")
    
    return results

def main():
    print("🌸 Training Models for Railway Deployment")
    print("="*50)
    
    # Create directories
    create_directories()
    
    # Load and prepare data
    df = load_and_prepare_data()
    X, y, label_encoder = preprocess_data(df)
    
    # Train and save models
    results = train_and_save_models(X, y)
    
    print("\n" + "="*50)
    print("🎉 Training Complete!")
    print("✅ All models saved to 'models/' directory")
    print("✅ Ready for Railway deployment")
    
    # List all created files
    print("\n📁 Created files:")
    for file in Path("models").glob("*.joblib"):
        print(f"   - {file}")

if __name__ == "__main__":
    main()
