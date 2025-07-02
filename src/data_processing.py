# src/data_processing.py

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_iv(df, feature, target):
    """Calculate Information Value for a feature."""
    if df[feature].dtype in ['int64', 'float64']:
        if df[feature].nunique() > 10:
            try:
                df[f'{feature}_binned'] = pd.qcut(df[feature], q=5, duplicates='drop', labels=False)
                bin_col = f'{feature}_binned'
            except ValueError:
                bin_col = feature
        else:
            bin_col = feature
    else:
        bin_col = feature

    crosstab = pd.crosstab(df[bin_col], df[target])
    if 0 not in crosstab.columns:
        crosstab[0] = 0
    if 1 not in crosstab.columns:
        crosstab[1] = 0

    crosstab['total'] = crosstab.sum(axis=1)
    crosstab['good_rate'] = (crosstab[0] + 1e-6) / (crosstab[0].sum() + 1e-6)
    crosstab['bad_rate'] = (crosstab[1] + 1e-6) / (crosstab[1].sum() + 1e-6)

    crosstab['woe'] = np.log(crosstab['good_rate'] / crosstab['bad_rate'])
    crosstab['iv'] = (crosstab['good_rate'] - crosstab['bad_rate']) * crosstab['woe']

    return crosstab['iv'].sum()


def calculate_iv_score(df, feature, target):
    return calculate_iv(df, feature, target)


class WOEEncoder:
    def __init__(self):
        self.woe_map = {}
        self.bins_map = {}

    def fit(self, X, y):
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                if X[col].nunique() > 10:
                    try:
                        X_binned, bins = pd.qcut(X[col], q=5, duplicates='drop', retbins=True, labels=False)
                        self.bins_map[col] = bins
                    except ValueError:
                        X_binned = X[col]
                        self.bins_map[col] = None
                else:
                    X_binned = X[col]
                    self.bins_map[col] = None
            else:
                X_binned = X[col]
                self.bins_map[col] = None

            crosstab = pd.crosstab(X_binned, y)
            if 0 not in crosstab.columns:
                crosstab[0] = 0
            if 1 not in crosstab.columns:
                crosstab[1] = 0

            crosstab['good_rate'] = (crosstab[0] + 1e-6) / (crosstab[0].sum() + 1e-6)
            crosstab['bad_rate'] = (crosstab[1] + 1e-6) / (crosstab[1].sum() + 1e-6)

            crosstab['woe'] = np.log(crosstab['good_rate'] / crosstab['bad_rate'])

            self.woe_map[col] = crosstab['woe'].to_dict()
        return self

    def transform(self, X):
        X_woe = X.copy()
        for col in X.columns:
            if col in self.woe_map:
                if self.bins_map.get(col) is not None:
                    try:
                        X_binned = pd.cut(X[col], bins=self.bins_map[col], include_lowest=True, labels=False)
                    except ValueError:
                        X_woe[f'{col}_woe'] = np.nan
                        continue
                else:
                    X_binned = X[col]

                X_woe[f'{col}_woe'] = X_binned.map(self.woe_map[col]).fillna(np.nan)
        return X_woe


def calculate_rfm_metrics(df):
    """Calculate Recency, Frequency, and Monetary metrics from raw transactions."""
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerId').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Amount': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerId', 'recency', 'frequency', 'monetary']
    return rfm


def validate_model_metrics(metrics_dict):
    """Check that expected model metrics are present and reasonable"""
    required_keys = ['accuracy', 'roc_auc', 'f1_score']
    for key in required_keys:
        if key not in metrics_dict:
            return False
        if not (0 <= metrics_dict[key] <= 1):
            return False
    return True
