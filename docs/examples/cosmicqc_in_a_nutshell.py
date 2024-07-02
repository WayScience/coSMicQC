# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # `coSMicQC` in a nutshell
#
# This notebook demonstrates various capabilities of `coSMicQC` using examples.

# +
import pathlib

import cosmicqc

# set a path for the parquet-based dataset
# (in this case, CellProfiler data processed by CytoTable)
data_path = (
    "../../tests/data/cytotable/NF1_cellpainting_data/Plate_2_with_image_data.parquet"
)

# set a context directory for images associated with the dataset
image_context_dir = pathlib.Path(data_path).parent / "Plate_2_images"

# create a cosmicqc SCDataFrame (single-cell DataFrame)
scdf = cosmicqc.SCDataFrame(data=data_path, data_context_dir=image_context_dir)

# display the dataframe
scdf

# +
# create a labeled dataset which includes z-scores and whether those scores
# are interpreted as outliers or inliers. We use pre-defined threshold sets
# loaded from a default configuration file (cosmicqc can accept user-defined files too!).
labeled_scdf = cosmicqc.analyze.label_outliers(
    df=scdf,
    include_threshold_scores=True,
)

# show the dataframe rows with only the last 8 columns (added from the label_outliers function)
labeled_scdf.iloc[:, -8:]
# -

# show histogram reports on the outliers and inliers for each threshold set in the new columns
labeled_scdf.show_report();

# show cropped images through SCDataFrame from the dataset to help analyze outliers
labeled_scdf.sort_values(by="cqc.large_nuclei.is_outlier", ascending=False)[
    [
        "Metadata_ImageNumber",
        "Metadata_Cells_Number_Object_Number",
        "cqc.large_nuclei.is_outlier",
        "Image_FileName_GFP",
        "Image_FileName_RFP",
        "Image_FileName_DAPI",
    ]
]
