"""
Used for downloading and preparing data for use
with specifically coSMicQC contamination detector tests.

This file may be processed using the following command from the root
of the project repository:
`poetry run python tests/data/cytotable/NF1_cellpainting_data/gather_contamination_data.py`
"""  # noqa: E501

import pathlib
import urllib.request

import pandas as pd

# Define the paths
test_data_path = "tests/data/cytotable/NF1_cellpainting_data/"
parquet_url = "https://github.com/WayScience/nf1_cellpainting_data/raw/main/3.processing_features/data/converted_data/Plate_3.parquet"
parquet_file_path = test_data_path + "Plate_3.parquet"
filtered_file_path = test_data_path + "Plate_3_filtered.parquet"

# Create the directory if it doesn't exist
pathlib.Path(test_data_path).mkdir(parents=True, exist_ok=True)

# Download the file if it doesn't already exist
if not pathlib.Path(parquet_file_path).exists():
    urllib.request.urlretrieve(parquet_url, parquet_file_path)

# Load the parquet file
nf1_df = pd.read_parquet(parquet_file_path)

# Filter rows where the well starts with B/C/D + ends with 1/5/9 (only 500 seed density)
filtered_nf1_df = nf1_df[nf1_df["Image_Metadata_Well"].str.match(r"^[BCD].*[159]$")]

# Save filtered DataFrame
filtered_nf1_df.to_parquet(filtered_file_path, index=False)

print(f"Filtered file saved to: {filtered_file_path}")

