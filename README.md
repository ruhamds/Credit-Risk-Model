# Credit Risk Modeling Project

This project implements a credit risk modeling pipeline using Python and MLflow, culminating in a FastAPI REST API deployment with Docker and CI/CD integration. The tasks progressively build the project from data engineering and model training to API development and containerization.

---

## Project Structure

credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                       # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here 
│   └── processed/             # Processed data for training
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Script for feature engineering
│   ├── train.py               # Script for model training
│   ├── predict.py             # Script for inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md

---

## Task Overview

### 📌 Task 1 - Credit Scoring Business Understanding

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

### Task 2: Data Profiling, Cleaning, and Exploratory Data Analysis (EDA)

- Load and profile dataset.
- Handle missing values and outliers.
- Perform feature exploration and visualization.
- Document data insights to guide feature engineering.

### Task 3: Feature Engineering Pipeline and Target Variable Creation

- Implement custom feature transformers in `data_processing.py` (e.g., RFM metrics).
- Create a proxy target variable for credit risk based on business rules.
- Apply Weight of Evidence (WOE) transformation and calculate Information Value (IV) for feature selection.

### Task 4: Model Training and Evaluation

- Train a Gradient Boosting model using engineered features.
- Implement SMOTE for class imbalance handling.
- Perform hyperparameter tuning.
- Log model and metrics with MLflow for experiment tracking.
- Visualize model performance with ROC, confusion matrix, etc.

### Task 5: API Development with FastAPI

- Create Pydantic models for input validation (`pydantic_models.py`).
- Develop prediction endpoint in FastAPI (`api/main.py`) that loads the MLflow model and returns risk predictions.
- Build modular inference logic (`predict.py`).
- Test API locally with Uvicorn.

### Task 6: Containerization and CI/CD Pipeline

- Write `Dockerfile` to containerize the FastAPI app.
- Optionally create `docker-compose.yml` for multi-service setups.
- Configure GitHub Actions (or similar) for linting, testing, and deployment automation.
- Use MLflow tracking URI and secrets management in the pipeline.
- Deploy and test containerized API in local or cloud environment.

---

## How to Run

### Local setup

1. Clone the repo:

git clone <repo-url>
cd credit-risk-model

2. Create and activate virtual environment:

python -m venv venv
.\venv\Scripts\activate  # Windows PowerShell
source venv/bin/activate # Linux/Mac

3. Install dependencies:

pip install -r requirements.txt

4. Run API locally:

uvicorn src.api.main:app --reload

Docker setup

1. Build Docker image:

docker build -t credit-risk-api .

2. Run container:

docker run -p 8000:8000 credit-risk-api

3. Access API at http://localhost:8000/docs.

Testing

pytest tests/test_data_processing.py -v

