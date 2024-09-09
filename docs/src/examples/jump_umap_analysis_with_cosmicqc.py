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

# + editable=true slideshow={"slide_type": ""}
import doctest
import pathlib
import shutil
from typing import List, Union

import duckdb
import holoviews
import hvplot.pandas
import numpy as np
import pandas as pd
import parsl
import pyarrow as pa
import pycytominer
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

# + editable=true slideshow={"slide_type": ""}
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
print("Large nuclei outliers:")
df_labeled_outliers = cosmicqc.analyze.find_outliers(
    df=df_merged_single_cells,
    metadata_columns=metadata_cols,
    feature_thresholds="large_nuclei",
)

# label outliers within the dataset
print("Elongated nuclei outliers:")
df_labeled_outliers = cosmicqc.analyze.find_outliers(
    df=df_merged_single_cells,
    metadata_columns=metadata_cols,
    feature_thresholds="elongated_nuclei",
)

# label outliers within the dataset
print("Small and low formfactor nuclei outliers:")
df_labeled_outliers = cosmicqc.analyze.find_outliers(
    df=df_merged_single_cells,
    metadata_columns=metadata_cols,
    feature_thresholds="small_and_low_formfactor_nuclei",
)

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

# subset the data
df_full_data = pa.Table.from_batches(
    [next(parquet.ParquetFile("./BR00117012.parquet").iter_batches(batch_size=60000))]
).to_pandas()
df_full_data

df_labeled_outliers["cqc.included_at_least_one_outlier"] = df_labeled_outliers[
    [col for col in df_labeled_outliers.columns.tolist() if ".is_outlier" in col]
].any(axis=1)
df_labeled_outliers["cqc.included_at_least_one_outlier"].value_counts()

# +
number_of_samples_per_group = 140000 // 2

# join feature selected data with outlier data
# using a sample based on outliers
# note: we do this to help ascertain outlier data for
# objects which may no longer include cosmicqc-checked columns
# (such as Nuclei_AreaShape_Area)
with duckdb.connect() as ddb:
    df_feature_selected_with_cqc_outlier_data = ddb.execute(
        f"""
        WITH sample_includes_outliers AS (
            SELECT feat_select.*, COLUMNS('cqc\..*')
            FROM read_parquet('./BR00117012.parquet') AS feat_select
            LEFT JOIN df_labeled_outliers AS cqc_outlier_data
            ON feat_select.Metadata_ImageNumber = cqc_outlier_data.Metadata_ImageNumber
            AND feat_select.Metadata_ObjectNumber = cqc_outlier_data.Metadata_ObjectNumber
            AND feat_select.Metadata_Plate = cqc_outlier_data.Metadata_Plate
            AND feat_select.Metadata_Well = cqc_outlier_data.Metadata_Well
            WHERE cqc_outlier_data."cqc.included_at_least_one_outlier" = TRUE
            USING SAMPLE {number_of_samples_per_group} ROWS
        ),
        sample_excludes_outliers AS (
            SELECT feat_select.*, COLUMNS('cqc\..*')
            FROM read_parquet('./BR00117012.parquet') AS feat_select
            LEFT JOIN df_labeled_outliers AS cqc_outlier_data
            ON feat_select.Metadata_ImageNumber = cqc_outlier_data.Metadata_ImageNumber
            AND feat_select.Metadata_ObjectNumber = cqc_outlier_data.Metadata_ObjectNumber
            AND feat_select.Metadata_Plate = cqc_outlier_data.Metadata_Plate
            AND feat_select.Metadata_Well = cqc_outlier_data.Metadata_Well
            WHERE cqc_outlier_data."cqc.included_at_least_one_outlier" = FALSE
            USING SAMPLE {number_of_samples_per_group} ROWS;
        )
        
        SELECT * FROM sample_includes_outliers
        
        UNION ALL

        SELECT * FROM sample_excludes_outliers
        """
    ).df()
df_feature_selected_with_cqc_outlier_data
# -

# normalize the data using pcytominer
df_pycytominer_normalized = pycytominer.normalize(
    profiles=df_full_data,
    features="infer",
    image_features=False,
    meta_features="infer",
    # samples="Metadata_control_type == 'negcon'",
    # should this be filled out?
    method="standardize",
    output_file=(parquet_pycytominer_normalized := "./BR00117012_normalized.parquet"),
    output_type="parquet",
)
df_pycytominer_normalized

# feature select normalized data using pycytominer
df_pycytominer_feature_selected = pycytominer.feature_select(
    profiles=parquet_pycytominer_normalized,
    operation=[
        "variance_threshold",
        "correlation_threshold",
        "blocklist",
        "drop_na_columns",
    ],
    na_cutoff=0,
    output_file=(
        parquet_pycytominer_feature_selected := "./BR00117012_feature_select.parquet"
    ),
    output_type="parquet",
)
df_pycytominer_feature_selected


# +
def generate_umap_embeddings(
    df_input: pd.DataFrame, cols_metadata: List[str]
) -> np.ndarray:
    """
    Generates UMAP (Uniform Manifold Approximation and Projection)
    embeddings for a given input dataframe,
    excluding specified metadata columns.

    Args:
        df_input (pd.DataFrame]):
            A dataframe which is expected to contain
            numeric columns to be used for UMAP fitting.
        cols_metadata (List[str]):
            A list of column names representing
            metadata columns that should be excluded
            from the UMAP transformation.

    Returns:
        np.ndarray:
            A dataframe containing the UMAP embeddings
            with 2 components for each row in the input.

    Doctests:
        >>> df = pd.DataFrame({
                'feature1': [1.0, 2.0, 3.0],
                'feature2': [4.0, 5.0, 6.0],
                'Metadata_Plate': ['plate1', 'plate2', 'plate3']
            })
        >>> metadata_cols = ['Metadata_Plate']
        >>> generate_umap_embeddings = generate_umap(df, metadata_cols)
        >>> type(umap_embeddings)
        np.ndarray
    """

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


all_embeddings = generate_umap_embeddings(
    df_input=df_feature_selected_with_cqc_outlier_data, cols_metadata=metadata_cols
)
all_embeddings


# +
def plot_hvplot_scatter_general(
    embeddings: np.ndarray,
) -> holoviews.core.spaces.DynamicMap:
    """
    Creates a generalized scatter hvplot for viewing
    UMAP embedding data.

    Args:
        embeddings (np.ndarray]):
            A numpy ndarray which includes
            embedding data to display.

    Returns:
        holoviews.core.spaces.DynamicMap:
            A dynamic holoviews scatter plot which may be
            displayed in a Jupyter notebook.
    """
    return pd.DataFrame(embeddings).hvplot.scatter(
        title="UMAP of JUMP dataset",
        x="0",
        y="1",
        alpha=0.1,
        rasterize=True,
        cnorm="linear",
        height=500,
        width=800,
    )


plot_hvplot_scatter_general(embeddings=embeddings)


# +
def plot_hvplot_scatter_outliers(
    embeddings: np.ndarray, cosmicqc_outlier_labels: pd.DataFrame, color_column: str
):
    """
    Creates an outlier-focused scatter hvplot for viewing
    UMAP embedding data with cosmicqc outliers coloration.

    Args:
        embeddings (np.ndarray]):
            A numpy ndarray which includes
            embedding data to display.
        cosmicqc_outlier_labels (pd.DataFrame):
            A dataframe which includes cosmicqc outlier
            data labels for use in coloration of plot.
        color_column (str):
            Column name from cosmicqc_outlier_labels to use
            for coloring the scatter plot.

    Returns:
        holoviews.core.spaces.DynamicMap:
            A dynamic holoviews scatter plot which may be
            displayed in a Jupyter notebook.
    """
    return pd.DataFrame(embeddings).hvplot.scatter(
        title="UMAP of JUMP erroneous outliers",
        x="0",
        y="1",
        alpha=0.1,
        rasterize=True,
        c=cosmicqc_outlier_labels[color_column].astype(int).values,
        cnorm="linear",
        cmap="plasma",
        bgcolor="black",
        height=500,
        width=800,
    )


plot_hvplot_scatter_outliers(
    embeddings=embeddings,
    cosmicqc_outlier_labels=df_feature_selected_with_cqc_outlier_data,
    color_column="cqc.small_and_low_formfactor_nuclei.is_outlier",
)
# -

plot_hvplot_scatter_outliers(
    embeddings=embeddings,
    cosmicqc_outlier_labels=df_feature_selected_with_cqc_outlier_data,
    color_column="cqc.elongated_nuclei.is_outlier",
)

plot_hvplot_scatter_outliers(
    embeddings=embeddings,
    cosmicqc_outlier_labels=df_feature_selected_with_cqc_outlier_data,
    color_column="cqc.large_nuclei.is_outlier",
)

with duckdb.connect() as ddb:
    df_feature_selected_without_cqc_outlier_data = ddb.execute(
        """
        SELECT *
        FROM df_feature_selected_with_cqc_outlier_data AS feat_select_ouliers
        WHERE feat_select_ouliers."cqc.small_and_low_formfactor_nuclei.is_outlier" = False
        AND feat_select_ouliers."cqc.large_nuclei.is_outlier" = False
        AND feat_select_ouliers."cqc.elongated_nuclei.is_outlier" = False
        """
    ).df()
df_feature_selected_without_cqc_outlier_data

embeddings_outliers_removed = generate_umap_embeddings(
    df_input=df_feature_selected_without_cqc_outlier_data, cols_metadata=metadata_cols
)
embeddings_outliers_removed

plot_hvplot_scatter_general(embeddings=embeddings_outliers_removed)


