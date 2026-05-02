import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.preprocess import get_train_test_split

def train_and_evaluate():
    """
    Trains a Random Forest model and evaluates it on the test set.
    Generates and saves performance plots.
    """
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = get_train_test_split()
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest_model.pkl')
    print("Model saved to models/random_forest_model.pkl")
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Ensure images directory exists
    os.makedirs('images', exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('images/confusion_matrix.png')
    plt.close()
    
    # 2. Feature Importance
    plt.figure(figsize=(10, 8))
    importances = model.feature_importances_
    features = X_train.columns
    indices = np.argsort(importances)
    
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('images/feature_importance.png')
    plt.close()
    
    print("Evaluation plots saved to images/ directory.")
    
if __name__ == "__main__":
    train_and_evaluate()
