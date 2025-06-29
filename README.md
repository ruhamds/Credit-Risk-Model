# 💳 Credit Risk Modeling Project

This project builds a credit risk model to predict **Probability of Default (PD)** using alternative data (e.g., transactions, product categories). It aligns with Basel II standards and includes a modular pipeline for EDA, feature engineering, and modeling.

---

## 📌 Task 1 - Credit Scoring Business Understanding

### 1. Basel II and Interpretability
The **Basel II Accord** requires transparent models for PD estimation. Interpretable models (e.g., Logistic Regression) and clear documentation ensure regulatory compliance and stakeholder trust.

### 2. Proxy for Default
We use `FraudResult` as a proxy for default due to missing direct labels. This supports modeling for thin-file borrowers but risks:
- **Inaccuracy**: Weak correlation with defaults.
- **Bias**: Unfair lending decisions.
- **Compliance**: Regulatory scrutiny.

### 3. Model Trade-offs
| Aspect             | Logistic Regression + WoE | Gradient Boosting |
|--------------------|---------------------------|-------------------|
| **Interpretability** | ✅ High                  | ❌ Low           |
| **Regulatory Fit**  | ✅ Audit-friendly         | ⚠️ Needs explanation |
| **Accuracy**        | ❌ May underfit          | ✅ High           |
| **Ease of Use**     | ✅ Simple                | ❌ Complex        |

Simple models are preferred for compliance; complex models need justification for accuracy.

---

## 🔍 Task 2 - Exploratory Data Analysis

EDA in `notebooks/1.0-eda.ipynb` analyzes a dataset of 95,662 transactions to guide feature engineering.

### Key Findings
- **Dataset**: 16 columns (e.g., `CustomerId`, `Amount`, `ProductCategory`, `FraudResult`).
- **Numerical**: `Amount` (mean: 6,718, std: 123,307, skewness: 51.1), `Value` (highly correlated with `Amount`: 0.99), `PricingStrategy`, `FraudResult` (imbalanced: 0.2% positive).
- **Categorical**: `ProductCategory` (top: financial_services, 47.5%), `ChannelId` (top: ChannelId_3, 59.5%).
- **Issues**: Missing `CountryCode`, `ProviderId`; outliers in `Amount` (30.9%), `Value` (14%).
- **Insights**: High skewness requires transformation; imbalanced `FraudResult` suggests sampling techniques.

---

## ⚙️ Task 3 - Feature Engineering Pipeline

The pipeline in `src/data_processing.py` prepares data for modeling, addressing EDA findings.

### Key Components
- **Aggregates**: `Amount_sum`, `Amount_mean`, `Amount_count`, `Amount_std` by `CustomerId`.
- **Date Features**: `transaction_hour`, `day`, `month`, `weekday` from `TransactionStartTime`.
- **Encoding**: One-hot encode `ProductCategory`, `ProviderId`, `ChannelId`.
- **Preprocessing**: Log-transform `Amount`, `Value`; median imputation for numericals; mode for categoricals; standardize numericals.
- **Output**: Saves `data/processed/transformed_credit_risk_data.csv`.

---

## 📁 Project Structure
\`\`\`
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
\`\`\`

## 🛠️ Setup
1. Clone repo:
   \`\`\`bash
   git clone <repository-url>
   cd credit-risk-model
   \`\`\`
2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
3. Add data to `data/raw/` (e.g., `credit_risk_data.csv`).
4. Run pipeline:
   \`\`\`bash
   python src/data_processing.py
   \`\`\`

## 🚀 Usage
- Run EDA: `notebooks/1.0-eda.ipynb`
- Train model: `python src/train.py`
- Predict: `python src/predict.py`
- Launch API: `uvicorn src.api.main:app --reload`

## 📋 Requirements
- Python 3.8+
- Libraries: pandas, sklearn, numpy, fastapi, uvicorn, flake8, pytest

## 🔜 Next Steps
- Implement `src/train.py` with Logistic Regression or Gradient Boosting.
- Add `src/predict.py` for inference.
- Write tests in `tests/test_data_processing.py`.
- Deploy API in `src/api/`
