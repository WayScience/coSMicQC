"""
Tests cosmicqc SCDataFrame module
"""

import pathlib

import pandas as pd
from cosmicqc.scdataframe import SCDataFrame
from pyarrow import parquet


def test_SCDataFrame_with_dataframe(
    tmp_path: pathlib.Path, basic_outlier_dataframe: pd.DataFrame
):
    """
    Tests SCDataFrame with pd.DataFrame input.
    """

    sc_df = SCDataFrame(data=basic_outlier_dataframe)

    # test that we ingested the data properly
    assert sc_df.data_source == "pd.DataFrame"
    assert sc_df.equals(basic_outlier_dataframe)

    # test export
    basic_outlier_dataframe.to_parquet(
        control_path := f"{tmp_path}/df_input_example.parquet"
    )
    sc_df.export(test_path := f"{tmp_path}/df_input_example1.parquet")

    assert parquet.read_table(control_path).equals(parquet.read_table(test_path))


def test_SCDataFrame_with_csv(tmp_path: pathlib.Path, basic_outlier_csv: str):
    """
    Tests SCDataFrame with CSV input.
    """

    sc_df = SCDataFrame(data=basic_outlier_csv)
    expected_df = pd.read_csv(basic_outlier_csv)

    # test that we ingested the data properly
    assert sc_df.data_source == basic_outlier_csv
    assert sc_df.equals(expected_df)

    # test export
    sc_df.export(test_path := f"{tmp_path}/df_input_example.csv", index=False)

    pd.testing.assert_frame_equal(expected_df, pd.read_csv(test_path))


def test_SCDataFrame_with_csv_gz(tmp_path: pathlib.Path, basic_outlier_csv_gz: str):
    """
    Tests SCDataFrame with CSV input.
    """

    sc_df = SCDataFrame(data=basic_outlier_csv_gz)
    expected_df = pd.read_csv(basic_outlier_csv_gz)

    # test that we ingested the data properly
    assert sc_df.data_source == basic_outlier_csv_gz
    assert sc_df.equals(expected_df)

    # test export
    sc_df.export(test_path := f"{tmp_path}/df_input_example.csv.gz", index=False)

    pd.testing.assert_frame_equal(
        expected_df, pd.read_csv(test_path, compression="gzip")
    )


def test_SCDataFrame_with_tsv(tmp_path: pathlib.Path, basic_outlier_tsv: str):
    """
    Tests SCDataFrame with TSV input.
    """

    sc_df = SCDataFrame(data=basic_outlier_tsv)
    expected_df = pd.read_csv(basic_outlier_tsv, delimiter="\t")

    # test that we ingested the data properly
    assert sc_df.data_source == basic_outlier_tsv
    assert sc_df.equals(expected_df)

    # test export
    sc_df.export(test_path := f"{tmp_path}/df_input_example.tsv", index=False)

    pd.testing.assert_frame_equal(expected_df, pd.read_csv(test_path, sep="\t"))


def test_SCDataFrame_with_parquet(tmp_path: pathlib.Path, basic_outlier_parquet: str):
    """
    Tests SCDataFrame with TSV input.
    """

    sc_df = SCDataFrame(data=basic_outlier_parquet)
    expected_df = pd.read_parquet(basic_outlier_parquet)

    # test that we ingested the data properly
    assert sc_df.data_source == basic_outlier_parquet
    assert sc_df.equals(expected_df)

    # test export
    sc_df.export(test_path := f"{tmp_path}/df_input_example2.parquet")

    assert parquet.read_table(basic_outlier_parquet).equals(
        parquet.read_table(test_path)
    )
