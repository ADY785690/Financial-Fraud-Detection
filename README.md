Financial Statement Fraud Detection System

Deep Learning | TensorFlow | SHAP Explainable AI | Streamlit

An AI-powered financial analysis application that uses Deep Learning and Explainable AI (SHAP) to identify potential anomalies in financial statements and support financial risk assessment.

Project Overview

Financial statement fraud can cause significant financial losses and damage investor confidence. This project aims to analyze financial data, identify suspicious patterns, and explain model predictions using SHAP (SHapley Additive exPlanations).

The application provides an interactive dashboard built with Streamlit.

Key Features

- Financial Data Explorer: Upload and explore financial statement datasets.
- Data Preprocessing: Handle missing values, prepare features, and scale numerical data.
- Deep Learning Model: Train a neural network for binary classification.
- Fraud Risk Prediction: Estimate potential fraud risk from financial indicators.
- SHAP Explainability: Understand which financial features influence model predictions.
- Interactive Dashboard: Explore results through visualizations and metrics.

Technology Stack

Technology| Purpose
Python| Core programming
Streamlit| Web application
Pandas & NumPy| Data processing
Scikit-learn| Preprocessing and evaluation
TensorFlow / Keras| Deep Learning
SHAP| Model explainability
Plotly| Interactive visualizations
Matplotlib| Data visualization

Project Structure

Financial-Fraud-Detection/
│
├── app.py
├── data_processor.py
├── requirements.txt
├── financialreport.csv
└── README.md

Installation & Setup

1. Clone the repository

git clone https://github.com/ADY785690/Financial-Fraud-Detection.git

2. Navigate to the project folder

cd Financial-Fraud-Detection

3. Install dependencies

pip install -r requirements.txt

4. Run the application

streamlit run app.py

The application will open in your browser.

Model Explainability

SHAP is used to interpret model predictions by estimating the contribution of individual financial features.

This helps users understand which financial indicators contribute to a model's risk assessment.

Dataset

The project uses financial statement data containing numerical financial indicators and derived financial ratios.

Note: If heuristic-generated fraud labels are used, they represent proxy anomaly labels and should not be interpreted as independently verified fraud cases.

Future Improvements

- Integration with real-world financial statement APIs.
- Support for multiple company financial reports.
- Advanced anomaly detection techniques.
- Improved model validation and performance.
- Automated financial risk monitoring.
- More comprehensive SHAP visualizations.

Disclaimer

This project is developed for educational, research, and financial risk-analysis purposes. Model predictions are indicative only and should not be treated as proof of financial fraud or professional audit conclusions.

Author
ADITYA KR SHARMA

Aditya K. Sharma

GitHub: "ADY785690" (https://github.com/ADY785690)
