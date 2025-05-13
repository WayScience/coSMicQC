"""
Used for downloading and preparing data for use
with specifically coSMicQC contamination detector tests.

This file may be processed using the following command from the root
of the project repository:
`poetry run python tests/data/cytotable/NF1_cellpainting_data/gather_contamination_data.py`
"""  # noqa: E501

import pathlib
import urllib.request

# Define the paths
test_data_path = "tests/data/cytotable/NF1_cellpainting_data/"
parquet_url = "https://github.com/WayScience/nf1_cellpainting_data/raw/main/3.processing_features/data/converted_data/Plate_3.parquet"
parquet_file_path = test_data_path + "Plate_3.parquet"

# Create the directory if it doesn't exist
pathlib.Path(test_data_path).mkdir(parents=True, exist_ok=True)

# Download the file if it doesn't already exist
if not pathlib.Path(parquet_file_path).exists():
    urllib.request.urlretrieve(parquet_url, parquet_file_path)
