"""
Tests cosmicqc qcdataframe module
"""

import pandas as pd
from cosmicqc.qcdataframe import QCDataFrame

def test_qcdataframe_init_with_dataframe(basic_outlier_dataframe: pd.DataFrame):
    """
    Tests QCDataFrame with pd.DataFrame input.
    """
    qc_df = QCDataFrame(data=basic_outlier_dataframe)
    assert qc_df.reference == "pd.DataFrame"
    assert qc_df.equals(basic_outlier_dataframe)

def test_qcdataframe_init_with_csv(basic_outlier_csv:str):
    """
    Tests QCDataFrame with CSV input.
    """
    qc_df = QCDataFrame(data=basic_outlier_csv)
    expected_df = pd.read_csv(basic_outlier_csv)
    assert qc_df.reference == basic_outlier_csv
    assert qc_df.equals(expected_df)

def test_qcdataframe_init_with_tsv(basic_outlier_tsv:str):
    """
    Tests QCDataFrame with TSV input.
    """
    qc_df = QCDataFrame(data=basic_outlier_tsv)
    expected_df = pd.read_csv(basic_outlier_tsv, delimiter='\t')
    assert qc_df.reference == basic_outlier_tsv
    assert qc_df.equals(expected_df)

def test_qcdataframe_init_with_parquet(basic_outlier_parquet:str):
    """
    Tests QCDataFrame with TSV input.
    """
    qc_df = QCDataFrame(data=basic_outlier_parquet)
    expected_df = pd.read_parquet(basic_outlier_parquet)
    assert qc_df.reference == basic_outlier_parquet
    assert qc_df.equals(expected_df)

