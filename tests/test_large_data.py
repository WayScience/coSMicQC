"""
Module dedicated to testing large data for cosmicqc
"""

import pytest
from cosmicqc import analyze
import cytotable
from pyarrow import parquet
import pathlib
import parsl
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
from pyarrow import parquet

@pytest.fixture(name="jump_cytotable_data")
def fixture_jump_cytotable_data(
    tmp_path: pathlib.Path,
):
    """
    Creates JUMP data processed through CytoTable as parquet for use in tests
    """

    s3_result = cytotable.convert(
        source_path=(
            "s3://cellpainting-gallery/cpg0016-jump/source_4/"
            "workspace/backend/2021_08_23_Batch12/BR00126114"
            "/BR00126114.sqlite"
        ),
        dest_path=(dest_path := f"{tmp_path}/BR00126114.parquet"),
        dest_datatype="parquet",
        source_datatype="sqlite",
        # set chunk size to amount which operates within
        # github actions runner images and related resource constraints.
        chunk_size=20000,
        preset="cellprofiler_sqlite_cpg0016_jump",
        sort_output=False,
        no_sign_request=True,
        # use explicit cache to avoid temp cache removal / overlaps with
        # sequential s3 SQLite files. See below for more information
        # https://cloudpathlib.drivendata.org/stable/caching/#automatically
        local_cache_dir=f"{tmp_path}/sqlite_s3_cache/2",
        parsl_config=parsl.load(
            Config(executors=[ThreadPoolExecutor(label="tpe_for_cosmicqc_testing")])
        ),
    )

    # read only the metadata from parquet file
    parquet_file_meta = parquet.ParquetFile(s3_result).metadata

    # check the shape of the data
    assert (parquet_file_meta.num_rows, parquet_file_meta.num_columns) == (74226, 5928)

    return dest_path


@pytest.mark.large_data_tests
def test_label_outliers_jump(
    jump_cytotable_data: str,
):
    """
    Test label_outliers with JUMP data
    """

    # test single-column result
    test_df = analyze.label_outliers(
        df=jump_cytotable_data,
        include_threshold_scores=True,
    )

    # check the shape
    assert test_df.shape == (74226, 5936)

    # check the detected outlier count
    assert test_df["cqc.small_and_low_formfactor_nuclei.is_outlier"].sum() == 1534  # noqa: PLR2004
    assert test_df["cqc.elongated_nuclei.is_outlier"].sum() == 3  # noqa: PLR2004
    assert test_df["cqc.large_nuclei.is_outlier"].sum() == 619  # noqa: PLR2004
