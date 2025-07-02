
import pytest
import pandas as pd
import numpy as np
from src.data_processing import calculate_rfm_metrics, calculate_iv_score, validate_model_metrics


class TestDataProcessing:

    def test_calculate_rfm_metrics(self):
        """Test RFM calculation function"""
        test_data = pd.DataFrame({
            'CustomerId': [1, 1, 2, 2, 3],
            'InvoiceDate': ['2023-01-01', '2023-01-15', '2023-01-10', '2023-01-20', '2023-01-05'],
            'InvoiceNo': ['A001', 'A002', 'A003', 'A004', 'A005'],
            'Amount': [100, 150, 200, 80, 300]
        })

        rfm_result = calculate_rfm_metrics(test_data)

        assert len(rfm_result) == 3
        assert 'recency' in rfm_result.columns
        assert 'frequency' in rfm_result.columns
        assert 'monetary' in rfm_result.columns

    def test_calculate_iv_score(self):
        """Test Information Value calculation"""
        np.random.seed(42)
        test_data = pd.DataFrame({
            'feature': np.random.normal(0, 1, 1000),
            'target': np.random.binomial(1, 0.3, 1000)
        })

        iv_score = calculate_iv_score(test_data, 'feature', 'target')

        assert isinstance(iv_score, float)
        assert iv_score >= 0
