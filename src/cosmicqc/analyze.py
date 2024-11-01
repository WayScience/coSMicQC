"""
Module for detecting various quality control aspects from source data.
"""

import operator
import pathlib
from functools import reduce
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml
from cytodataframe.frame import CytoDataFrame
from scipy.stats import zscore as scipy_zscore

DEFAULT_QC_THRESHOLD_FILE = (
    f"{pathlib.Path(__file__).parent!s}/data/qc_nuclei_thresholds_default.yml"
)


def identify_outliers(
    df: Union[CytoDataFrame, pd.DataFrame, str],
    feature_thresholds: Union[Dict[str, float], str],
    feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
    include_threshold_scores: bool = False,
    export_path: Optional[str] = None,
) -> Union[pd.Series, CytoDataFrame]:
    """
    This function uses z-scoring to format the data for detecting outlier
    nuclei or cells using specific CellProfiler features. Thresholds are
    the number of standard deviations away from the mean, either above
    (positive) or below (negative). We recommend making sure to not use a
    threshold of 0 as that would represent the whole dataset.

    Args:
        df: Union[CytoDataFrame, pd.DataFrame, str]
            DataFrame or file string-based filepath of a
            Parquet, CSV, or TSV file with CytoTable output or similar data.
        feature_thresholds: Dict[str, float]
            One of two options:
            A dictionary with the feature name(s) as the key(s) and their assigned
            threshold for identifying outliers. Positive int for the threshold
            will detect outliers "above" than the mean, negative int will detect
            outliers "below" the mean.
            Or a string which is a named key reference found within
            the feature_thresholds_file yaml file.
        feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
            An optional feature thresholds file where thresholds may be
            defined within a file.
        include_threshold_scores: bool
            Whether to include the threshold scores in addition to whether
            the threshold set passes per row.
        export_path: Optional[str] = None
            An optional path to export the data using CytoDataFrame export
            capabilities. If None no export is performed.
            Note: compatible exports are CSV's, TSV's, and parquet.

    Returns:
        Union[pd.Series, CytoDataFrame]:
            Outlier series with booleans based on whether outliers were detected
            or not for use within other functions.
    """

    # interpret the df as CytoDataFrame
    df = CytoDataFrame(data=df)

    # create a copy of the dataframe to ensure
    # we don't modify the supplied dataframe inplace.
    outlier_df = df.copy()

    thresholds_name = (
        f"cqc.{feature_thresholds}"
        if isinstance(feature_thresholds, str)
        else "cqc.custom"
    )

    if isinstance(feature_thresholds, str):
        feature_thresholds = read_thresholds_set_from_file(
            feature_thresholds=feature_thresholds,
            feature_thresholds_file=feature_thresholds_file,
        )

    # Create z-score columns for each feature to reference during outlier detection
    zscore_columns = {}
    for feature in feature_thresholds:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' does not exist in the DataFrame.")
        outlier_df[(colname := f"{thresholds_name}.Z_Score.{feature}")] = scipy_zscore(
            df[feature]
        )
        zscore_columns[feature] = colname

    # Create outlier detection conditions for each feature
    conditions = []
    for feature, threshold in feature_thresholds.items():
        # For positive thresholds, look for outliers that are
        # that number of std "above" the mean
        if threshold > 0:
            condition = outlier_df[zscore_columns[feature]] > threshold
        # For negative thresholds, look for outliers that are
        # that number of std "below" the mean
        else:
            condition = outlier_df[zscore_columns[feature]] < threshold
        conditions.append(condition)

    result = (
        # create a boolean pd.series identifier for dataframe
        # based on all conditions for use within other functions.
        reduce(operator.and_, conditions)
        if not include_threshold_scores
        # otherwise, provide the threshold zscore col and the above column
        else CytoDataFrame(
            data=pd.concat(
                [
                    # grab only the outlier zscore columns from the outlier_df
                    outlier_df[zscore_columns.values()],
                    CytoDataFrame(
                        {
                            f"{thresholds_name}.is_outlier": reduce(
                                operator.and_, conditions
                            )
                        }
                    ),
                ],
                axis=1,
            ),
            data_context_dir=df._custom_attrs["data_context_dir"],
            data_mask_context_dir=df._custom_attrs["data_mask_context_dir"],
        )
    )

    if export_path is not None:
        if isinstance(result, pd.Series):
            CytoDataFrame(result).export(file_path=export_path)
        else:
            result.export(file_path=export_path)

    return result


def find_outliers(
    df: Union[CytoDataFrame, pd.DataFrame, str],
    metadata_columns: List[str],
    feature_thresholds: Union[Dict[str, float], str],
    feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
    export_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    This function uses identify_outliers to return a dataframe
    with only the outliers and provided metadata columns.

    Args:
        df: Union[CytoDataFrame, pd.DataFrame, str]
            DataFrame or file string-based filepath of a
            Parquet, CSV, or TSV file with CytoTable output or similar data.
        metadata_columns: List[str]
            List of metadata columns that should be outputted with the outlier data.
        feature_thresholds: Dict[str, float]
            One of two options:
            A dictionary with the feature name(s) as the key(s) and their assigned
            threshold for identifying outliers. Positive int for the threshold
            will detect outliers "above" than the mean, negative int will detect
            outliers "below" the mean.
            Or a string which is a named key reference found within
            the feature_thresholds_file yaml file.
        feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
            An optional feature thresholds file where thresholds may be
            defined within a file.
        export_path: Optional[str] = None
            An optional path to export the data using CytoDataFrame export
            capabilities. If None no export is performed.
            Note: compatible exports are CSV's, TSV's, and parquet.

    Returns:
        pd.DataFrame:
            Outlier data frame for the given conditions.
    """

    # interpret the df as CytoDataFrame
    df = CytoDataFrame(data=df)

    if isinstance(feature_thresholds, str):
        feature_thresholds = read_thresholds_set_from_file(
            feature_thresholds=feature_thresholds,
            feature_thresholds_file=feature_thresholds_file,
        )

    # Filter DataFrame for outliers using all conditions
    outliers_df = df[
        # use identify outliers as a mask on the full dataframe
        identify_outliers(
            df=df,
            feature_thresholds=feature_thresholds,
            feature_thresholds_file=feature_thresholds_file,
        )
    ]

    # Print outliers count and range for each feature
    print(
        "Number of outliers:",
        outliers_df.shape[0],
        f"({'{:.2f}'.format((outliers_df.shape[0] / df.shape[0])*100)}%)",
    )
    print("Outliers Range:")
    for feature in feature_thresholds:
        print(f"{feature} Min:", outliers_df[feature].min())
        print(f"{feature} Max:", outliers_df[feature].max())

    # Include metadata columns in the output DataFrame
    columns_to_include = list(feature_thresholds.keys()) + metadata_columns

    result = outliers_df[columns_to_include]

    # export the file if specified
    if export_path is not None:
        result.export(file_path=export_path)

    # Return outliers DataFrame with specified columns
    return result


def label_outliers(  # noqa: PLR0913
    df: Union[CytoDataFrame, pd.DataFrame, str],
    feature_thresholds: Optional[Union[Dict[str, float], str]] = None,
    feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
    include_threshold_scores: bool = False,
    export_path: Optional[str] = None,
    report_path: Optional[str] = None,
    **kwargs: Dict[str, Any],
) -> CytoDataFrame:
    """
    Use identify_outliers to label the original dataset for
    where a cell passed or failed the quality control condition(s).

        Args:
            df: Union[CytoDataFrame, pd.DataFrame, str]
                DataFrame or file string-based filepath of a
                Parquet, CSV, or TSV file with CytoTable output or similar data.
            feature_thresholds: Dict[str, float]
                One of two options:
                A dictionary with the feature name(s) as the key(s) and their assigned
                threshold for identifying outliers. Positive int for the threshold
                will detect outliers "above" than the mean, negative int will detect
                outliers "below" the mean.
                Or a string which is a named key reference found within
                the feature_thresholds_file yaml file.
            feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
                An optional feature thresholds file where thresholds may be
                defined within a file.
            include_threshold_scores: bool = False
                Whether to include the scores in addition to whether an outlier
                was detected or not.
            export_path: Optional[str] = None
                An optional path to export the data using CytoDataFrame export
                capabilities. If None no export is performed.
                Note: compatible exports are CSV's, TSV's, and parquet.

        Returns:
            CytoDataFrame:
                Full dataframe with optional scores and outlier boolean column.
    """

    # interpret the df as CytoDataFrame
    df = CytoDataFrame(data=df)

    # for single outlier processing
    if isinstance(feature_thresholds, (str, dict)):
        # return the outlier dataframe for one threshold rule
        identified_outliers = identify_outliers(
            df=df,
            feature_thresholds=feature_thresholds,
            feature_thresholds_file=feature_thresholds_file,
            include_threshold_scores=include_threshold_scores,
        )

        result = CytoDataFrame(
            data=pd.concat(
                [
                    df,
                    (
                        identified_outliers
                        if isinstance(identified_outliers, pd.DataFrame)
                        else CytoDataFrame(
                            {
                                (
                                    f"cqc.{feature_thresholds}.is_outlier"
                                    if isinstance(feature_thresholds, str)
                                    else "cqc.custom.is_outlier"
                                ): identified_outliers
                            }
                        )
                    ),
                ],
                axis=1,
            ),
            data_context_dir=df._custom_attrs["data_context_dir"],
            data_mask_context_dir=df._custom_attrs["data_mask_context_dir"],
        )

    # for multiple outlier processing
    elif feature_thresholds is None:
        # return the outlier dataframe for all threshold rules
        labeled_df = pd.concat(
            [df]
            + [
                # identify outliers for each threshold rule
                identify_outliers(
                    df=df,
                    feature_thresholds=thresholds,
                    feature_thresholds_file=feature_thresholds_file,
                    include_threshold_scores=include_threshold_scores,
                )
                # loop through each threshold rule
                for thresholds in read_thresholds_set_from_file(
                    feature_thresholds_file=feature_thresholds_file,
                )
            ],
            axis=1,
        )
        # return a dataframe with a deduplicated columns by name
        result = CytoDataFrame(
            labeled_df.loc[:, ~labeled_df.columns.duplicated()],
            data_context_dir=df._custom_attrs["data_context_dir"],
            data_mask_context_dir=df._custom_attrs["data_mask_context_dir"],
        )

    # export the file if specified
    if export_path is not None:
        result.export(file_path=export_path)

    # if we have a report path, generate the report and use kwargs
    if report_path is not None:
        result.show_report(report_path=report_path, **kwargs)

    return result


def read_thresholds_set_from_file(
    feature_thresholds_file: str, feature_thresholds: Optional[str] = None
) -> Union[Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Reads a set of feature thresholds from a specified file.

    This function takes the path to a feature thresholds file and a
    specific feature threshold string, reads the file, and returns
    the thresholds set from the file.

    Args:
        feature_thresholds_file (str):
            The path to the file containing feature thresholds.
        feature_thresholds (Optional str, default None):
            A string specifying the feature thresholds.
            If we have None, return all thresholds.

    Returns:
        dict: A dictionary containing the processed feature thresholds.

    Raises:
        LookupError: If the file does not contain the specified feature_thresholds key.
    """

    # open the yaml file
    with open(feature_thresholds_file, "r") as file:
        thresholds = yaml.safe_load(file)

    # if no feature thresholds name is specified, return all thresholds
    if feature_thresholds is None:
        return thresholds["thresholds"]

    if feature_thresholds not in thresholds["thresholds"]:
        raise LookupError(
            (
                f"Unable to find threshold set by name {feature_thresholds}"
                f" within {feature_thresholds_file}"
            )
        )

    return thresholds["thresholds"][feature_thresholds]
