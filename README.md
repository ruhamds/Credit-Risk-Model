💳 Credit Risk Modeling Project
This project builds a credit risk scoring model to predict Probability of Default (PD) using alternative data (e.g., transactions, product categories). It aligns with Basel II standards and includes a modular pipeline for feature engineering, EDA, and modeling.

📌 Task 1 - Credit Scoring Business Understanding
1. Basel II and Interpretability
The Basel II Accord requires transparent, auditable models for estimating PD. Interpretable models (e.g., Logistic Regression) and clear documentation (e.g., feature engineering) ensure regulatory compliance and stakeholder trust.
2. Proxy for Default
Lacking a direct default label, we use FraudResult as a proxy. This enables modeling for thin-file borrowers but risks:

Inaccuracy: Poor correlation with defaults.
Bias: Unfair lending decisions.
Compliance: Regulatory scrutiny over proxy validity.

3. Model Trade-offs


| Aspect | Simple (Logistic Regression + WoE) | Complex (e.g., Random Forest, XGBoost) |
|--------|------------------------------------|----------------------------------------|
| **Interpretability** | ✅ High – easy to explain | ❌ Low – black box |
| **Regulatory Acceptance** | ✅ Strong | ⚠️ Conditional |
| **Performance** | ❌ Limited | ✅ High |
| **Training Cost** | ✅ Low | ❌ Higher |
| **Use Case Fit** | Ideal for compliance and explanation | Ideal for performance and optimization |


Simple models are preferred for compliance; complex models offer better accuracy but require justification.

🔍 Task 2 - Exploratory Data Analysis
EDA in notebooks/1.0-eda.ipynb analyzes the dataset to guide feature engineering.
Key Steps

Dataset: 16 columns (e.g., CustomerId, Amount, ProductCategory, FraudResult).
Analysis: Summary stats, histograms for Amount, frequency of ProductCategory, correlations, missing values, outliers.
Insights: Skewed Amount, imbalanced ProductCategory, missing CountryCode suggest preprocessing needs.


⚙️ Task 3 - Feature Engineering Pipeline
The pipeline in src/data_processing.py prepares transaction data for modeling.
Key Components

Aggregates: Amount_sum, Amount_mean, Amount_count, Amount_std by CustomerId.
Date Features: transaction_hour, day, month, year from TransactionStartTime.
Encoding: One-hot encode ProductCategory, ProviderId, ChannelId.
Preprocessing: Median imputation for numericals, mode for categoricals, standardize numericals.
Output: Saves data/processed/transformed_credit_risk_data.csv.


📁 Project Structure
credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD pipeline
├── data/                         # Raw/processed data (ignored by Git)
│   ├── raw/                     # e.g., credit_risk_data.csv
│   └── processed/               # e.g., transformed_credit_risk_data.csv
├── notebooks/                    # Exploratory analysis
│   └── 1.0-eda.ipynb            # EDA notebook
├── src/                         # Source code
│   ├── __init__.py
│   ├── data_processing.py       # Feature engineering
│   ├── train.py                # Model training
│   ├── predict.py              # Inference
│   └── api/                    # FastAPI app
│       ├── main.py
│       └── pydantic_models.py
├── tests/                       # Unit tests
│   └── test_data_processing.py
├── Dockerfile                   # Docker config
├── docker-compose.yml           # Docker Compose
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore
└── README.md                    # Documentation

🛠️ Setup

Clone repo:git clone <https://github.com/ruhamds/Credit-Risk-Model.git>


Install dependencies:pip install -r requirements.txt


Add data to data/raw/ (e.g., credit_risk_data.csv).
Run pipeline:python src/data_processing.py



🚀 Usage

Run EDA: notebooks/1.0-eda.ipynb
Train model: python src/train.py
Predict: python src/predict.py
Launch API: uvicorn src.api.main:app --reload

📋 Requirements

Python 3.8+
Libraries: pandas, sklearn, numpy, fastapi, uvicorn, flake8, pytest

🔜 Next Steps

Implement src/train.py for model training.
Add src/predict.py for inference.
Write tests in tests/test_data_processing.py.
Deploy API in src/api/.
