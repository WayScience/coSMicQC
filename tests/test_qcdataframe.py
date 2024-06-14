"""
Tests cosmicqc SCDataFrame module
"""

import pandas as pd
from cosmicqc.scdataframe import SCDataFrame


def test_SCDataFrame_init_with_dataframe(basic_outlier_dataframe: pd.DataFrame):
    """
    Tests SCDataFrame with pd.DataFrame input.
    """
    sc_df = SCDataFrame(data=basic_outlier_dataframe)
    assert sc_df.data_source == "pd.DataFrame"
    assert sc_df.equals(basic_outlier_dataframe)


def test_SCDataFrame_init_with_csv(basic_outlier_csv: str):
    """
    Tests SCDataFrame with CSV input.
    """
    sc_df = SCDataFrame(data=basic_outlier_csv)
    expected_df = pd.read_csv(basic_outlier_csv)
    assert sc_df.data_source == basic_outlier_csv
    assert sc_df.equals(expected_df)


def test_SCDataFrame_init_with_csv_gz(basic_outlier_csv_gz: str):
    """
    Tests SCDataFrame with CSV input.
    """
    sc_df = SCDataFrame(data=basic_outlier_csv_gz)
    expected_df = pd.read_csv(basic_outlier_csv_gz)
    assert sc_df.data_source == basic_outlier_csv_gz
    assert sc_df.equals(expected_df)


def test_SCDataFrame_init_with_tsv(basic_outlier_tsv: str):
    """
    Tests SCDataFrame with TSV input.
    """
    sc_df = SCDataFrame(data=basic_outlier_tsv)
    expected_df = pd.read_csv(basic_outlier_tsv, delimiter="\t")
    assert sc_df.data_source == basic_outlier_tsv
    assert sc_df.equals(expected_df)


def test_SCDataFrame_init_with_parquet(basic_outlier_parquet: str):
    """
    Tests SCDataFrame with TSV input.
    """
    sc_df = SCDataFrame(data=basic_outlier_parquet)
    expected_df = pd.read_parquet(basic_outlier_parquet)
    assert sc_df.data_source == basic_outlier_parquet
    assert sc_df.equals(expected_df)
