"""
Used for downloading and preparing data for use
with coSMicQC tests.
"""

import pathlib
import subprocess
import zipfile

import pandas as pd

# Define the paths
test_data_path = "tests/data/cytotable/NF1_cellpainting_data/"
sqlite_url = "https://github.com/WayScience/nf1_cellpainting_data/raw/main/2.cellprofiler_analysis/analysis_output/Plate_2/Plate_2_nf1_analysis.sqlite"
sqlite_file_path = test_data_path + "Plate_2_nf1_analysis.sqlite"
parquet_url = "https://github.com/WayScience/nf1_cellpainting_data/raw/main/3.processing_features/data/converted_data/Plate_2.parquet"
parquet_file_path = test_data_path + "Plate_2.parquet"
image_zip_url = "https://figshare.com/ndownloader/articles/22233700/versions/4"
image_zip_file_path = test_data_path + "Plate_2_images.zip"
image_extract_dir = test_data_path + "Plate_2_images"
joined_data_path = test_data_path + "Plate_2_with_image_data.parquet"

for url, file_path in zip(
    [sqlite_url, parquet_url, image_zip_url],
    [sqlite_file_path, parquet_file_path, image_zip_file_path],
):
    if not pathlib.Path(file_path).is_file():
        print(f"Downloading {file_path}...")
        subprocess.run(["wget", "-O", file_path, url], check=True)
        print(f"Downloaded {file_path}")

# Check if the zip file exists
if not pathlib.Path(image_zip_file_path).is_file():
    print(f"{image_zip_file_path} does not exist.")
else:
    # Create the extraction directory if it doesn't exist
    if not pathlib.Path(image_extract_dir).is_dir():
        pathlib.Path(image_extract_dir).mkdir(parents=True)

    # Extract the zip file
    with zipfile.ZipFile(image_zip_file_path, "r") as zip_ref:
        zip_ref.extractall(image_extract_dir)

    print(f"Extracted {image_zip_file_path} to {image_extract_dir}")

# form parquet table which includes image paths
df_cytotable = pd.read_parquet(parquet_file_path)

df_image = pd.read_sql(
    sql="SELECT * FROM per_image;", con=f"sqlite:///{sqlite_file_path}"
)

df_full = pd.merge(
    left=df_cytotable,
    right=df_image,
    how="left",
    left_on="Metadata_ImageNumber",
    right_on="ImageNumber",
)


def modify_filename(file_path):
    return file_path.replace(
        "_illumcorrect.tiff", ".tif"
    )

df_full["Image_URL_DAPI"] = df_full["Image_URL_DAPI"].apply(modify_filename)
df_full["Image_URL_GFP"] = df_full["Image_URL_GFP"].apply(modify_filename)
df_full["Image_URL_RFP"] = df_full["Image_URL_RFP"].apply(modify_filename)
df_full["Image_FileName_DAPI"] = df_full["Image_FileName_DAPI"].apply(modify_filename)
df_full["Image_FileName_GFP"] = df_full["Image_FileName_GFP"].apply(modify_filename)
df_full["Image_FileName_RFP"] = df_full["Image_FileName_RFP"].apply(modify_filename)

df_full.to_parquet(joined_data_path)
