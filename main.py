import os
from src.data_generator import generate_synthetic_data
from src.train_model import train_and_evaluate
from src.simulate import simulate_new_customers

def main():
    print("=" * 50)
    print(" CUSTOMER CHURN PREDICTION PIPELINE ")
    print("=" * 50)
    
    # Ensure directories exist
    for dir_name in ['data', 'models', 'outputs', 'images']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Step 1: Data Generation
    print("\n[Step 1] Generating Synthetic Data...")
    generate_synthetic_data(num_samples=5000)
    
    # Step 2: Model Training & Evaluation
    print("\n[Step 2] Training and Evaluating Model...")
    train_and_evaluate()
    
    # Step 3: Virtual Simulation
    print("\n[Step 3] Running Virtual Simulation...")
    simulate_new_customers()
    
    print("\n" + "=" * 50)
    print(" PIPELINE COMPLETED SUCCESSFULLY!")
    print("Check the 'images' folder for visualizations and 'models' for the saved model.")
    print("=" * 50)

if __name__ == "__main__":
    main()
