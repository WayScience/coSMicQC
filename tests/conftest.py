"""
Fixtures for testing via pytest.
See here for more information:
https://docs.pytest.org/en/7.1.x/explanation/fixtures.html
"""

import pathlib

import cosmicqc
import pandas as pd
import pytest


@pytest.fixture(name="cytotable_CFReT_data_df")
def fixture_cytotable_CFReT_df():
    """
    Return df to test CytoTable CFReT_data
    """
    return pd.read_parquet(
        "tests/data/cytotable/CFRet_data/test_localhost231120090001_converted.parquet"
    )


@pytest.fixture(name="basic_outlier_dataframe")
def fixture_basic_outlier_dataframe():
    """
    Creates basic example data for use in tests
    """
    return pd.DataFrame({"example_feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})


@pytest.fixture(name="basic_outlier_csv")
def fixture_basic_outlier_csv(
    tmp_path: pathlib.Path, basic_outlier_dataframe: pd.DataFrame
):
    """
    Creates basic example data csv for use in tests
    """

    basic_outlier_dataframe.to_csv(
        csv_path := tmp_path / "basic_example.csv", index=False
    )

    return csv_path


@pytest.fixture(name="basic_outlier_csv_gz")
def fixture_basic_outlier_csv_gz(
    tmp_path: pathlib.Path, basic_outlier_dataframe: pd.DataFrame
):
    """
    Creates basic example data csv for use in tests
    """

    basic_outlier_dataframe.to_csv(
        csv_gz_path := tmp_path / "example.csv.gz", index=False, compression="gzip"
    )

    return csv_gz_path


@pytest.fixture(name="basic_outlier_tsv")
def fixture_basic_outlier_tsv(
    tmp_path: pathlib.Path, basic_outlier_dataframe: pd.DataFrame
):
    """
    Creates basic example data tsv for use in tests
    """

    basic_outlier_dataframe.to_csv(
        tsv_path := tmp_path / "example.tsv", sep="\t", index=False
    )

    return tsv_path


@pytest.fixture(name="basic_outlier_parquet")
def fixture_basic_outlier_parquet(
    tmp_path: pathlib.Path, basic_outlier_dataframe: pd.DataFrame
):
    """
    Creates basic example data parquet for use in tests
    """

    basic_outlier_dataframe.to_parquet(
        parquet_path := tmp_path / "example.parquet", index=False
    )

    return parquet_path


def test_generate_show_report_html_output(cytotable_CFReT_data_df: pd.DataFrame):
    """
    Used for generating report output for use with other tests.
    """

    df = cosmicqc.analyze.label_outliers(
        df=cytotable_CFReT_data_df,
        include_threshold_scores=True,
    )

    df.show_report(
        report_path=pathlib.Path(__file__).parent
        / "data"
        / "coSMicQC"
        / "show_report"
        / "cosmicqc_example_report.html"
    )
