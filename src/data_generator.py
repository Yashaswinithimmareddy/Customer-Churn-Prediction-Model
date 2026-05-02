import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=5000, output_path='data/customer_churn_data.csv'):
    """
    Generates a realistic synthetic customer churn dataset.
    """
    np.random.seed(42)
    
    # 1. Generate features
    customer_ids = [f'CUST_{i:05d}' for i in range(1, num_samples + 1)]
    genders = np.random.choice(['Male', 'Female'], num_samples)
    senior_citizen = np.random.choice([0, 1], num_samples, p=[0.85, 0.15])
    partner = np.random.choice(['Yes', 'No'], num_samples)
    dependents = np.random.choice(['Yes', 'No'], num_samples)
    
    # Tenure in months (1 to 72)
    tenure = np.random.randint(1, 73, num_samples)
    
    # Services
    phone_service = np.random.choice(['Yes', 'No'], num_samples, p=[0.9, 0.1])
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], num_samples, p=[0.35, 0.45, 0.20])
    
    # Contract and Billing
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples, p=[0.55, 0.20, 0.25])
    paperless_billing = np.random.choice(['Yes', 'No'], num_samples)
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], num_samples)
    
    # Financials
    monthly_charges = []
    for internet in internet_service:
        if internet == 'Fiber optic':
            monthly_charges.append(np.random.uniform(70, 115))
        elif internet == 'DSL':
            monthly_charges.append(np.random.uniform(45, 80))
        else:
            monthly_charges.append(np.random.uniform(18, 25))
    monthly_charges = np.array(monthly_charges)
    
    # Total charges is mostly tenure * monthly_charges, with some noise
    total_charges = monthly_charges * tenure * np.random.uniform(0.95, 1.05, num_samples)
    
    # 2. Determine Churn (Target Variable)
    # Customers are more likely to churn if: Month-to-month contract, High monthly charges, Low tenure, Fiber Optic
    churn_prob = np.zeros(num_samples)
    
    # Base churn probability
    churn_prob += 0.05
    
    # Add conditions
    churn_prob[contract == 'Month-to-month'] += 0.25
    churn_prob[contract == 'Two year'] -= 0.15
    churn_prob[tenure < 12] += 0.20
    churn_prob[tenure > 48] -= 0.10
    churn_prob[internet_service == 'Fiber optic'] += 0.15
    churn_prob[payment_method == 'Electronic check'] += 0.10
    
    # Clip probabilities between 0 and 1
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Generate actual churn labels based on probability
    churn = np.random.binomial(1, churn_prob)
    churn_labels = ['Yes' if c == 1 else 'No' for c in churn]
    
    # 3. Create DataFrame
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Gender': genders,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'Tenure': tenure,
        'PhoneService': phone_service,
        'InternetService': internet_service,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': np.round(monthly_charges, 2),
        'TotalCharges': np.round(total_charges, 2),
        'Churn': churn_labels
    })
    
    # Introduce some missing values in TotalCharges to simulate real-world data issues
    missing_indices = np.random.choice(num_samples, int(num_samples * 0.005), replace=False)
    df.loc[missing_indices, 'TotalCharges'] = np.nan
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset with {num_samples} records saved to {output_path}")
    
    return df

if __name__ == "__main__":
    generate_synthetic_data()
