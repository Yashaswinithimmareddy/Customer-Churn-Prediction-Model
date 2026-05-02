import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_and_preprocess_data(filepath='data/customer_churn_data.csv', is_training=True):
    """
    Loads raw data, handles missing values, encodes categories, and scales numericals.
    Saves the preprocessors (scalers, encoders) if is_training is True.
    """
    df = pd.read_csv(filepath)
    
    # 1. Handle Missing Values
    # In pandas, empty strings in numeric columns might be read as objects. Let's force to numeric.
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill missing with median
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # 2. Drop unnecessary columns
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)
        
    # Separate features and target
    if 'Churn' in df.columns:
        X = df.drop('Churn', axis=1)
        y = df['Churn'].map({'Yes': 1, 'No': 0})
    else:
        X = df.copy()
        y = None
        
    # 3. Encoding Categorical Variables
    # We will use One-Hot Encoding for Nominal variables and Label Encoding for Binary
    binary_cols = ['Gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    multi_cols = ['InternetService', 'Contract', 'PaymentMethod']
    
    # Label encode binary
    le_dict = {}
    if is_training:
        for col in binary_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                le_dict[col] = le
        os.makedirs('models', exist_ok=True)
        joblib.dump(le_dict, 'models/label_encoders.pkl')
    else:
        le_dict = joblib.load('models/label_encoders.pkl')
        for col in binary_cols:
            if col in X.columns:
                # Handle unseen labels by assigning a default or using a robust mapper (simplified here)
                X[col] = X[col].map(lambda s: 1 if s in ['Yes', 'Female'] else 0)

    # One-Hot encode multi-class
    X = pd.get_dummies(X, columns=multi_cols, drop_first=True)
    
    # Ensure all columns exist in test data that existed in training data
    if is_training:
        joblib.dump(X.columns.tolist(), 'models/model_columns.pkl')
    else:
        model_cols = joblib.load('models/model_columns.pkl')
        # Add missing dummy columns filled with 0
        for col in model_cols:
            if col not in X.columns:
                X[col] = 0
        # Ensure order matches
        X = X[model_cols]
    
    # 4. Feature Scaling
    num_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    if is_training:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        joblib.dump(scaler, 'models/scaler.pkl')
    else:
        scaler = joblib.load('models/scaler.pkl')
        X[num_cols] = scaler.transform(X[num_cols])
        
    return X, y

def get_train_test_split():
    """
    Returns train and test sets, applying SMOTE to training data to handle imbalance.
    """
    X, y = load_and_preprocess_data(is_training=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE to training data only
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    print(f"Original training shape: {X_train.shape}, Churn count: {y_train.sum()}")
    print(f"Resampled training shape: {X_train_sm.shape}, Churn count: {y_train_sm.sum()}")
    
    return X_train_sm, X_test, y_train_sm, y_test

if __name__ == "__main__":
    get_train_test_split()
