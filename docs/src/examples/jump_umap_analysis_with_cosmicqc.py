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

# set bokeh for visualizations with hvplot
hvplot.extension("bokeh")

pathlib.Path("./images").mkdir(exist_ok=True)

# +
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

df_labeled_outliers["analysis.included_at_least_one_outlier"] = df_labeled_outliers[
    [col for col in df_labeled_outliers.columns.tolist() if ".is_outlier" in col]
].any(axis=1)
outliers_counts = df_labeled_outliers[
    "analysis.included_at_least_one_outlier"
].value_counts()
outliers_counts

print(
    (outliers_counts.iloc[1] / outliers_counts.iloc[0]) * 100,
    "%",
    "of",
    outliers_counts.iloc[0],
    "include erroneous outliers of some kind.",
)

# show histograms to help visualize the data
df_labeled_outliers.show_report();

# +
sample_fraction = 0.44

df_features = pa.Table.from_batches(
    [next(parquet.ParquetFile("./BR00117012.parquet").iter_batches(batch_size=10000))]
).to_pandas()

# group by metadata_well for all features then sample
# the dataset by a fraction.
df_features = (
    df_features.groupby(["Metadata_Well"])[df_features.columns]
    .apply(lambda x: x.sample(frac=sample_fraction))
    .reset_index(drop=True)
)

# join the sampled feature data with the cosmicqc outlier data
df_feature_selected_with_cqc_outlier_data = df_features.merge(
    df_labeled_outliers,
    how="inner",
    left_on=[
        "Metadata_ImageNumber",
        "Metadata_ObjectNumber",
        "Metadata_Plate",
        "Metadata_Well",
    ],
    right_on=[
        "Metadata_ImageNumber",
        "Metadata_ObjectNumber",
        "Metadata_Plate",
        "Metadata_Well",
    ],
)

df_feature_selected_with_cqc_outlier_data
# -

# show our data value counts regarding outliers vs inliers
df_feature_selected_with_cqc_outlier_data[
    "analysis.included_at_least_one_outlier"
].value_counts()

# prepare data for normalization and feature selection
# by removing cosmicqc and analaysis focused columns.
df_for_normalize_and_feature_select = df_feature_selected_with_cqc_outlier_data[
    [
        col
        for col in df_feature_selected_with_cqc_outlier_data.columns.tolist()
        if "cqc." not in col and "analysis." not in col
    ]
]
# show the modified column count
len(df_for_normalize_and_feature_select.columns)

# normalize the data using pcytominer
df_pycytominer_normalized = pycytominer.normalize(
    profiles=df_for_normalize_and_feature_select,
    features="infer",
    image_features=False,
    meta_features="infer",
    method="standardize",
    output_file=(parquet_pycytominer_normalized := "./BR00117012_normalized.parquet"),
    output_type="parquet",
)

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


embeddings_with_outliers = generate_umap_embeddings(
    df_input=pd.read_parquet(parquet_pycytominer_feature_selected),
    cols_metadata=metadata_cols,
)
embeddings_with_outliers


# +
def plot_hvplot_scatter_general(
    embeddings: np.ndarray, filename: str
) -> holoviews.core.spaces.DynamicMap:
    """
    Creates a generalized scatter hvplot for viewing
    UMAP embedding data.

    Args:
        embeddings (np.ndarray]):
            A numpy ndarray which includes
            embedding data to display.
        filename (str):
            Filename which indicates where to export the
            plot.

    Returns:
        holoviews.core.spaces.DynamicMap:
            A dynamic holoviews scatter plot which may be
            displayed in a Jupyter notebook.
    """
    
    # build a scatter plot through hvplot
    plot = pd.DataFrame(embeddings).hvplot.scatter(
        title="UMAP of JUMP dataset BR00117012.sqlite",
        x="0",
        y="1",
        alpha=0.1,
        rasterize=True,
        cnorm="linear",
        height=500,
        width=800,
    )

    # export the plot
    hvplot.save(obj=plot, filename=filename)
    
    return plot


plot_hvplot_scatter_general(embeddings=embeddings_with_outliers, filename="./images/umap_BR00117012.png")


# +
def plot_hvplot_scatter_outliers(
    embeddings: np.ndarray,
    cosmicqc_outlier_labels: pd.DataFrame,
    color_column: str,
    title: str,
    filename: str,
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
        title (str):
            Title for the UMAP scatter plot.
        filename (str):
            Filename which indicates where to export the
            plot.

    Returns:
        holoviews.core.spaces.DynamicMap:
            A dynamic holoviews scatter plot which may be
            displayed in a Jupyter notebook.
    """

    # build a scatter plot through hvplot
    plot = pd.DataFrame(embeddings).hvplot.scatter(
        title=title,
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

    # export the plot
    hvplot.save(obj=plot, filename=filename)

    return plot


plot_hvplot_scatter_outliers(
    embeddings=embeddings_with_outliers,
    cosmicqc_outlier_labels=df_feature_selected_with_cqc_outlier_data,
    color_column="analysis.included_at_least_one_outlier",
    title="UMAP of JUMP erroneous outliers within BR00117012.sqlite",
    filename="./images/umap_erroneous_outliers_BR00117012.png",
)
# -

plot = plot_hvplot_scatter_outliers(
    embeddings=embeddings_with_outliers,
    cosmicqc_outlier_labels=df_feature_selected_with_cqc_outlier_data,
    color_column="cqc.small_and_low_formfactor_nuclei.is_outlier",
    title="UMAP of JUMP small and low formfactor nuclei outliers within BR00117012.sqlite",
    filename="./images/umap_small_and_low_formfactor_nuclei_outliers_BR00117012.png",
)

plot = plot_hvplot_scatter_outliers(
    embeddings=embeddings_with_outliers,
    cosmicqc_outlier_labels=df_feature_selected_with_cqc_outlier_data,
    color_column="cqc.elongated_nuclei.is_outlier",
    title="UMAP of JUMP elongated nuclei outliers within BR00117012.sqlite",
    filename="./images/umap_elongated_nuclei_outliers_BR00117012.png",
)

plot_hvplot_scatter_outliers(
    embeddings=embeddings_with_outliers,
    cosmicqc_outlier_labels=df_feature_selected_with_cqc_outlier_data,
    color_column="cqc.large_nuclei.is_outlier",
    title="UMAP of JUMP large nuclei outliers within BR00117012.sqlite",
    filename="./images/umap_large_nuclei_outliers_BR00117012.png",
)