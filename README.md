# 📉 Customer Churn Prediction Model

This project demonstrates an end-to-end Machine Learning pipeline to predict customer churn. It covers everything from generating synthetic, industry-realistic customer data, to handling imbalanced classes using SMOTE, training a Random Forest Classifier, and interpreting business insights through Feature Importance.

## 🎯 Business Objective
Customer churn occurs when customers stop doing business with a company. Acquiring new customers is 5x to 25x more expensive than retaining existing ones. By accurately predicting which customers are at risk of churning, a business can:
- Target retention campaigns and special offers effectively.
- Increase Customer Lifetime Value (LTV) and overall revenue.
- Identify the key drivers (e.g., Contract Type, Tenure) that lead to churn.

## 🧠 Tech Stack
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Random Forest)
- **Imbalance Handling:** Imbalanced-Learn (SMOTE)
- **Visualization:** Matplotlib, Seaborn

## 🗂 Project Architecture & Workflow
1. **Data Generation:** Simulated customer data with demographics, account information, and service usage.
2. **Preprocessing:** Handled missing values, encoded categorical variables (Label Encoding & One-Hot Encoding), and scaled numerical features.
3. **Imbalance Handling:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the churn vs. no-churn classes in the training set.
4. **Model Training:** Trained a **Random Forest Classifier** to predict the likelihood of churn.
5. **Evaluation:** Evaluated the model using Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.
6. **Simulation:** Built a virtual simulation script to ingest "new" customers and output a churn risk probability.

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the complete pipeline:**
   ```bash
   python main.py
   ```
   *This single command will generate the data, train the model, save visualizations, and run the virtual simulation.*

## 📊 Results & Business Insights
Check the `images/` folder after running the pipeline for visualizations:
- **Feature Importance:** Identifies the top factors driving churn (often Contract Type and Tenure).
- **Confusion Matrix:** Shows the true positives, false positives, etc.

## 📂 Folder Structure
```
Customer-Churn-Prediction/
├── data/                  # Generated synthetic dataset
├── images/                # Visualizations (Feature Importance, Confusion Matrix)
├── models/                # Saved ML models and preprocessors (.pkl)
├── src/                   # Core Python scripts
│   ├── data_generator.py  # Generates realistic synthetic data
│   ├── preprocess.py      # Cleans, encodes, and scales data
│   ├── train_model.py     # Trains the Random Forest model
│   └── simulate.py        # Simulates predictions for new customers
├── main.py                # Master script to run the full pipeline
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```
