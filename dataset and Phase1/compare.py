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

# Load and prepare data
df = pd.read_csv("Iris.csv")
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "scaler.joblib")

# Define models and their parameter grids
model_params = {
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [10, 50, 100],
            "max_depth": [None, 3, 5]
        }
    },
    "SVM": {
        "model": SVC(),
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
        "model": DecisionTreeClassifier(),
        "params": {
            "max_depth": [None, 3, 5],
            "criterion": ["gini", "entropy"]
        }
    },
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=200),
        "params": {
            "C": [0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"]
        }
    }
}

best_models = {}

# Train and hypertune models
for name, mp in model_params.items():
    clf = GridSearchCV(mp["model"], mp["params"], cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test)

    print(f"\n{name} (Best Params: {clf.best_params_})")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model
    joblib.dump(best_model, f"{name}_model.joblib")
