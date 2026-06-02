🛡️ FraudShield AI — Credit Card Fraud Detection System


An AI-powered web application that detects credit card fraud in real-time using a Random Forest ML model — featuring secure login, bulk CSV analysis, live risk scoring, and an interactive analytics dashboard.


📌 Problem Statement
Credit card fraud is a critical financial threat. With only 0.17% of transactions being fraudulent in real-world data, standard models fail due to severe class imbalance. FraudShield AI solves this using advanced statistical techniques and ensemble learning to achieve near-perfect detection accuracy.

Features

🔐 Secure Login / Register System — user authentication with hashed credentials
⚡ Real-Time Fraud Prediction — instant risk assessment for individual transactions
📊 Risk Score Gauge — visual confidence meter showing fraud probability
📂 Bulk CSV Upload — analyse thousands of transactions at once
📈 Live Analytics Dashboard — transaction trends, fraud distribution charts
🔔 Instant Alerts — flagged transactions highlighted with risk level


ML Pipeline & Methodology
Raw Data (2,84,807 transactions)
        ↓
Exploratory Data Analysis (EDA)
        ↓
Feature Scaling (StandardScaler)
        ↓
Class Imbalance Handling (SMOTE)
        ↓
Model Training (Random Forest Classifier)
        ↓
Evaluation (ROC-AUC, Precision, Recall, F1)
        ↓
Streamlit Deployment
Key Statistics
MetricValueDataset Size2,84,807 transactionsFraud Rate (original)0.17%ModelRandom Forest (100 estimators)ROC-AUC Score99.99%False NegativesNear-zero
Why SMOTE?
The dataset had extreme class imbalance — only 492 fraudulent transactions out of 284,807. Training directly on this data causes the model to simply predict "not fraud" every time and still be 99.8% accurate. SMOTE (Synthetic Minority Oversampling Technique) generates synthetic fraud samples to balance the distribution, forcing the model to actually learn fraud patterns.

Tech Stack
LayerTechnologyLanguagePython 3.10+ML LibraryScikit-learnData ProcessingNumPy, PandasVisualizationPlotly, MatplotlibClass Balancingimbalanced-learn (SMOTE)Web AppStreamlitAuthYAML-based config with hashing

Project Structure
fraud-detection/
│
├── app.py                  # Main Streamlit application
├── config.yaml             # User credentials config
├── requirements.txt        # Dependencies
├── .gitignore
│
├── model/
│   └── fraud_model.pkl     # Trained Random Forest model
│
└── notebooks/
    └── fraud_analysis.ipynb  # EDA + Model training notebook

Getting Started
Prerequisites

Python 3.10 or above
pip

Installation
Step 1 — Clone the repository
bashgit clone https://github.com/arpitpandey291105-coder/fraud-detection.git
cd fraud-detection
Step 2 — Create a virtual environment (recommended)
bashpython -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
Step 3 — Install dependencies
bashpip install -r requirements.txt
Step 4 — Run the app
bashstreamlit run app.py
Step 5 — Open in browser
http://localhost:8501

📷 Screenshots

<img width="1894" height="895" alt="Screenshot 2026-06-02 122210" src="https://github.com/user-attachments/assets/e38a346f-34b9-4f81-8347-911e4fe4eb4b" />
<img width="1890" height="899" alt="Screenshot 2026-06-02 122305" src="https://github.com/user-attachments/assets/0ecd6879-041d-44de-ba04-e711d4b05cfa" />
<img width="1905" height="883" alt="Screenshot 2026-06-02 122454" src="https://github.com/user-attachments/assets/1c20204e-909a-4506-abf1-48025b54b91f" />
<img width="1901" height="888" alt="Screenshot 2026-06-02 122743" src="https://github.com/user-attachments/assets/18ca19c6-7aa9-45d0-a6e6-fb2a13aa8f58" />
<img width="1902" height="879" alt="Screenshot 2026-06-02 122848" src="https://github.com/user-attachments/assets/b43ed9ca-6d04-4bff-bdd6-81a87788eaf9" />
<img width="1902" height="893" alt="Screenshot 2026-06-02 122917" src="https://github.com/user-attachments/assets/35883f5f-31b4-4eae-abd2-cee778edca47" />

Model Performance
The Random Forest model was evaluated using multiple metrics to ensure robustness beyond simple accuracy:

ROC-AUC: 99.99% — near-perfect ability to distinguish fraud from genuine transactions
Precision & Recall — optimised to minimise false negatives (missing a fraud is more costly than a false alarm)
Confusion Matrix — near-zero false negatives on the test set


Dataset
This project uses the Credit Card Fraud Detection dataset from Kaggle (ULB Machine Learning Group).

284,807 transactions by European cardholders (September 2013)
Features V1–V28 are PCA-transformed for confidentiality
Features: Time, Amount, Class (0 = genuine, 1 = fraud)

Author
Arpit Pandey
B.Tech CSE (AI) — Noida Institute of Engineering and Technology

📧 arpitpandey291105@gmail.com
🏆 ET AI Hackathon 2026 — Semi-Finalist & Top Performing Candidate
💻 LeetCode: 260+ problems solved | 100 Days Badge


License
This project is licensed under the MIT License — see the LICENSE file for details.

If you found this project useful, please give it a star!
