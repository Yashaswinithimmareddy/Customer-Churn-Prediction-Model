import pandas as pd
import joblib
import warnings
from src.preprocess import load_and_preprocess_data

warnings.filterwarnings('ignore')

def simulate_new_customers():
    """
    Creates hypothetical new customers, runs them through the pipeline, 
    and predicts their churn probability.
    """
    print("\n--- Running Virtual Simulation ---")
    # Define a couple of hypothetical customers
    # Customer 1: High risk (Month-to-month, Fiber optic, High charges, Low tenure)
    # Customer 2: Low risk (Two year, No internet, Low charges, High tenure)
    
    new_data = pd.DataFrame({
        'Gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 0],
        'Partner': ['No', 'Yes'],
        'Dependents': ['No', 'Yes'],
        'Tenure': [2, 65],  # low vs high tenure
        'PhoneService': ['Yes', 'Yes'],
        'InternetService': ['Fiber optic', 'No'],
        'Contract': ['Month-to-month', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Credit card'],
        'MonthlyCharges': [95.50, 20.00],
        'TotalCharges': [191.00, 1300.00]
    })
    
    # Save temporarily to use the preprocessing pipeline
    temp_path = 'data/temp_simulation.csv'
    new_data.to_csv(temp_path, index=False)
    
    # Load and preprocess using the saved artifacts
    X_new, _ = load_and_preprocess_data(temp_path, is_training=False)
    
    # Load model
    model = joblib.load('models/random_forest_model.pkl')
    
    # Predict
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]
    
    print("\nSimulation Results:")
    for i, prob in enumerate(probabilities):
        risk_level = "HIGH RISK" if predictions[i] == 1 else "LOW RISK"
        print(f"Customer {i+1} ({new_data.iloc[i]['Contract']} contract, {new_data.iloc[i]['Tenure']} months tenure):")
        print(f"  -> Prediction: {risk_level} (Probability of Churn: {prob*100:.2f}%)")
        print("-" * 40)

if __name__ == "__main__":
    simulate_new_customers()
