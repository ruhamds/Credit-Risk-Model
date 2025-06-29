Credit Risk Probability Model for Alternative Data
Overview
This project develops a machine learning model to predict the Probability of Default (PD) for borrowers using alternative data (e.g., transaction amounts, product categories like airtime or utility bills). The model aligns with Basel II standards and uses a structured pipeline for feature engineering, training, and inference. The project leverages a transaction-level dataset with features like CustomerId, Amount, ProductCategory, and FraudResult (used as a proxy for default).
Project Structure
credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD configuration
├── data/                         # Raw and processed data (ignored by Git)
│   ├── raw/                     # Raw datasets (e.g., credit_risk_data.csv)
│   └── processed/               # Processed datasets (e.g., transformed_credit_risk_data.csv)
├── notebooks/                    # Exploratory analysis
│   └── 1.0-eda.ipynb            # EDA notebook
├── src/                         # Source code
│   ├── __init__.py
│   ├── data_processing.py       # Feature engineering pipeline
│   ├── train.py                # Model training script
│   ├── predict.py              # Inference script
│   └── api/                    # FastAPI application
│       ├── main.py
│       └── pydantic_models.py
├── tests/                       # Unit tests
│   └── test_data_processing.py
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose configuration
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # Project documentation

Setup

Clone the repository:git clone <repository-url>
cd credit-risk-model


Install dependencies:pip install -r requirements.txt


Place raw data in data/raw/ (e.g., credit_risk_data.csv).
Run feature engineering:python src/data_processing.py



Usage

Run EDA in notebooks/1.0-eda.ipynb.
Train the model with python src/train.py.
Make predictions with python src/predict.py.
Launch the API with uvicorn src.api.main:app --reload.

Requirements

Python 3.8+
Libraries: pandas, sklearn, numpy, fastapi, uvicorn, flake8, pytest

Task 1: Credit Scoring Business Understanding
This task focuses on understanding credit risk and its implications for model development, drawing on key references like the Basel II Accord and alternative data guidelines.
How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
The Basel II Accord (https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf) emphasizes robust risk measurement through its Internal Ratings-Based (IRB) approach, requiring banks to estimate Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). This mandates interpretable models to ensure regulators can validate risk assessments. Well-documented models, with clear feature engineering (e.g., aggregating transaction amounts) and transparent logic, are essential for compliance, enabling audits and ensuring alignment with capital requirements. In our project, using alternative data like transaction amounts and product categories, interpretability ensures stakeholders understand how features (e.g., Amount_sum) predict PD, supporting Basel II’s transparency requirements.
Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
Our dataset uses FraudResult as a proxy for default, as direct default labels are unavailable. A proxy variable, like FraudResult or frequency of late transactions, is necessary to approximate creditworthiness, especially with alternative data (e.g., ProductCategory like utility bills), as noted in the HKMA reference (https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf). This enables modeling for "thin-file" borrowers lacking traditional credit histories. However, business risks include:

Inaccuracy: If FraudResult poorly correlates with actual defaults, predictions may be unreliable, leading to incorrect lending decisions.
Bias: Proxies like transaction patterns may inadvertently favor certain demographics, risking unfair lending practices.
Regulatory Scrutiny: Regulators may question the proxy’s validity, requiring robust validation (e.g., statistical tests) to ensure compliance.

What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
Logistic Regression with Weight of Evidence (WoE) is interpretable, aligning with Basel II’s transparency requirements, as it clearly shows how features (e.g., Amount_mean) impact PD. It’s easier to explain to regulators and stakeholders, as described in the Towards Data Science article (https://towards将是
Task 2: Exploratory Data Analysis (EDA)
The EDA process, conducted in notebooks/1.0-eda.ipynb, explores the transaction-level dataset to uncover patterns, identify data quality issues, and guide feature engineering. Key steps include:

Dataset Structure: Analyzed the dataset’s structure (e.g., 16 columns including CustomerId, Amount, `Product daqueles

System: Dataset Structure:

Contains 16 columns: TransactionId, BatchId, AccountId, SubscriptionId, CustomerId, CurrencyCode, CountryCode, ProviderId, ProductId, ProductCategory, ChannelId, Amount, Value, TransactionStartTime, PricingStrategy, FraudResult.
Numerical columns: Amount, Value, PricingStrategy.
Categorical columns: CurrencyCode, ProviderId, ProductCategory, ChannelId.
Target variable: FraudResult (proxy for default).

Key Steps

Summary Statistics: Computed central tendency, dispersion, and distribution shape for numerical features (e.g., Amount, Value).
Numerical Distributions: Visualized histograms and KDE plots to identify skewness (e.g., Amount may be right-skewed) and outliers.
Categorical Distributions: Analyzed frequency of categories in ProductCategory (e.g., airtime, financial_services) and ChannelId.
Correlation Analysis: Examined relationships between numerical features (e.g., Amount vs. FraudResult) using a correlation matrix heatmap.
Missing Values: Identified missing data in CountryCode, ProviderId, AccountId, and Value, suggesting imputation strategies (e.g., median for numerical, mode for categorical).
Outlier Detection: Used box plots to detect outliers in Amount and Value.

Hypothetical Insights

Skewed Transaction Amounts: Amount is right-skewed, suggesting log-transformation for modeling.
Imbalanced Product Categories: ProductCategory shows dominance of certain types (e.g., airtime), requiring balancing techniques.
Missing Data Patterns: Missing CountryCode and ProviderId values may indicate data quality issues, necessitating imputation.
Correlation with FraudResult: Features like Amount and PricingStrategy may show moderate correlations with FraudResult, guiding feature selection.
Outliers: Extreme Amount values (e.g., large transactions) may require capping or robust modeling.

These insights inform feature engineering by prioritizing predictive features, handling skewness, and addressing missing data.
Task 3: Feature Engineering
The feature engineering pipeline, implemented in src/data_processing.py, transforms raw transaction data into a model-ready format using sklearn.pipeline.Pipeline. Key steps include:

Aggregate Features: Created customer-level features:
Amount_sum: Total transaction amount per CustomerId.
Amount_mean: Average transaction amount.
Amount_count: Number of transactions.
Amount_std: Standard deviation of transaction amounts.


Extracted Features: Derived temporal features from TransactionStartTime:
transaction_hour, transaction_day, transaction_month, transaction_year.


Categorical Encoding: Applied one-hot encoding to ProductCategory, ProviderId, and ChannelId.
Missing Value Imputation: Imputed numerical features (e.g., Amount) with median and categorical features with mode.
Standardization: Scaled numerical features to mean 0 and standard deviation 1.
Output: Generated data/processed/transformed_credit_risk_data.csv with model-ready features.

The pipeline ensures reproducibility and aligns with Basel II’s transparency requirements by documenting transformations clearly.
Next Steps

Complete model training in src/train.py using Logistic Regression or Gradient Boosting.
Implement inference logic in src/predict.py.
Develop unit tests in tests/test_data_processing.py to validate the pipeline.
Deploy the FastAPI application in src/api/ for real-time predictions.
