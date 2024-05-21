"""
Fixtures for testing via pytest.
See here for more information:
https://docs.pytest.org/en/7.1.x/explanation/fixtures.html
"""

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
