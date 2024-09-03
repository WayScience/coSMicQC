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

# # JUMP UMAP analysis with coSMicQC
#
# This notebook analyzes [JUMP](https://jump-cellpainting.broadinstitute.org/) data by leveraging [UMAP](https://arxiv.org/abs/1802.03426) and [coSMicQC](https://github.com/WayScience/coSMicQC).
#
# ## Outline
#
# We focus on a single file from the JUMP dataset: [`BR00117012.sqlite`](https://open.quiltdata.com/b/cellpainting-gallery/tree/cpg0000-jump-pilot/source_4/workspace/backend/2020_11_04_CPJUMP1/BR00117012/BR00117012.sqlite).
# This file is downloaded and prepared by [CytoTable](https://github.com/cytomining/CytoTable) to form a single-cell [Parquet](https://parquet.apache.org/) file which includes all compartment feature data.
# We use coSMicQC to find and remove erroneous outlier data in order to prepare for UMAP analysis.
# Afterwards, we use UMAP to demonstrate patterns within the data.
#

# +
import pathlib
import shutil
from typing import List, Union

import hvplot.pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parsl
import pyarrow as pa
import umap
from cytotable.convert import convert
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
from pyarrow import parquet

import cosmicqc

# check if we already have prepared data
if not pathlib.Path("./BR00117012.parquet").is_file():
    # process BR00117012.sqlite using CytoTable to prepare data
    merged_single_cells = convert(
        source_path=(
            "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace"
            "/backend/2020_11_04_CPJUMP1/BR00117012/BR00117012.sqlite"
        ),
        dest_path="./BR00117012.parquet",
        dest_datatype="parquet",
        source_datatype="sqlite",
        chunk_size=8000,
        preset="cellprofiler_sqlite_cpg0016_jump",
        no_sign_request=True,
        # use explicit cache to avoid temp cache removal
        local_cache_dir="./sqlite_s3_cache/",
        parsl_config=Config(
            executors=[ThreadPoolExecutor(label="tpe_for_jump_processing")]
        ),
        sort_output=False,
    )
else:
    merged_single_cells = "./BR00117012.parquet"

# read only the metadata from parquet file
parquet.ParquetFile(merged_single_cells).metadata
# -

schema = parquet.read_schema(merged_single_cells)
print("\n".join(str(schema).split("\n")[:12]))

# +
metadata_cols = [
    "Metadata_ImageNumber",
    "Image_Metadata_Row",
    "Image_Metadata_Site",
    "Metadata_ObjectNumber",
    "Metadata_Plate",
    "Metadata_Well",
    "Image_TableNumber",
]

df_merged_single_cells = pd.read_parquet(
    path=merged_single_cells,
    columns=metadata_cols
    + [
        "Nuclei_AreaShape_Area",
        "Nuclei_AreaShape_FormFactor",
        "Nuclei_AreaShape_Eccentricity",
    ],
)
df_merged_single_cells.head()
# -

# label outliers within the dataset
df_labeled_outliers = cosmicqc.analyze.label_outliers(
    df=df_merged_single_cells,
    include_threshold_scores=True,
)
# show added columns
df_labeled_outliers[
    [col for col in df_labeled_outliers.columns.tolist() if "cqc." in col]
].head()

# show histograms to help visualize the data
df_labeled_outliers.show_report();


# +
def generate_umap(
    df_input: Union[cosmicqc.CytoDataFrame, pd.DataFrame], cols_metadata: List[str]
) -> Union[cosmicqc.CytoDataFrame, pd.DataFrame]:

    # Set constants
    umap_random_seed = 0
    umap_n_components = 2

    # Make sure to reinitialize UMAP instance per plate
    umap_fit = umap.UMAP(
        # random_state=umap_random_seed,
        n_components=umap_n_components
    )

    # Fit UMAP and convert to pandas DataFrame
    embeddings = umap_fit.fit_transform(
        X=df_input[
            [
                col
                for col in df_input.columns.tolist()
                if col not in metadata_cols and "cqc." not in col
            ]
        ].select_dtypes(include=[np.number])
    )

    return embeddings


df_labeled = cosmicqc.analyze.label_outliers(
    df=(
        pa.Table.from_batches(
            [
                next(
                    parquet.ParquetFile("./BR00117012.parquet").iter_batches(
                        batch_size=60000
                    )
                )
            ]
        )
        .to_pandas()
        .replace([np.inf, -np.inf], np.nan)
        .pipe(lambda df: df.drop(labels=df.columns[df.isna().any()], axis=1))
        .reset_index()
    ),
    include_threshold_scores=True,
)

embeddings = generate_umap(df_input=df_labeled, cols_metadata=metadata_cols)
embeddings
# -

pd.DataFrame(embeddings).hvplot.scatter(
    title="UMAP of JUMP dataset",
    x="0",
    y="1",
    alpha=0.1,
    rasterize=True,
    cnorm="linear",
    height=500,
    width=800,
)

pd.DataFrame(embeddings).hvplot.scatter(
    title="UMAP of JUMP erroneous outliers",
    x="0",
    y="1",
    alpha=0.1,
    rasterize=True,
    c=df_labeled["cqc.small_and_low_formfactor_nuclei.is_outlier"].astype(int).values,
    cnorm="linear",
    cmap="plasma",
    bgcolor="black",
    height=500,
    width=800,
)


