"""
# README

## 📌 Project: Train a Supervised Learning Model with Hyperparameter Tuning

### 🔹 Dataset:
Cleaned Titanic dataset (`cleaned_titanic.csv`) generated from Task 2.

### 🔹 Objective:
- Train a supervised classification model on Titanic survival prediction.
- Apply hyperparameter tuning (GridSearchCV).
- Evaluate model performance using accuracy, precision, recall, F1, confusion matrix.
- Save trained model for reuse.

### 🔹 Deliverables:
1. `train_model.py` (this script)
2. `best_model.pkl` (saved trained model)
3. Evaluation metrics and confusion matrix plot
4. `requirements.txt`

### 🔹 Libraries Used:
- pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

"""

# ================================
# 📌 Import Libraries
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# ================================
# 📌 Function: Load Data
# ================================
def load_data(path):
    data = pd.read_csv(path)

    # Separate features and target
    X = data.drop('survived', axis=1)
    y = data['survived']

    # Ensure all categorical columns are encoded to numeric
    X = pd.get_dummies(X, drop_first=True)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ================================
# 📌 Function: Train Base Model
# ================================
def train_base_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nBase Model Performance:")
    evaluate_model(y_test, y_pred)
    return model

# ================================
# 📌 Function: Hyperparameter Tuning
# ================================
def tune_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, 
                               cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print("\nBest Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# ================================
# 📌 Function: Evaluate Model
# ================================
def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# ================================
# 📌 Main Execution
# ================================
if __name__ == "__main__":
    # Load preprocessed dataset
    X_train, X_test, y_train, y_test = load_data("cleaned_titanic.csv")

    # Train base model
    base_model = train_base_model(X_train, y_train, X_test, y_test)

    # Hyperparameter tuning
    tuned_model = tune_model(X_train, y_train)

    # Evaluate tuned model
    y_pred_tuned = tuned_model.predict(X_test)
    print("\nTuned Model Performance:")
    evaluate_model(y_test, y_pred_tuned)

    # Save best model
    joblib.dump(tuned_model, "best_model.pkl")
    print("\n✅ Best model saved as 'best_model.pkl'")