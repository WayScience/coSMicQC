"""
Module for detecting various quality control aspects from source data.
"""

import operator
import pathlib
from functools import reduce
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml
from scipy.stats import zscore as scipy_zscore

DEFAULT_QC_THRESHOLD_FILE = (
    f"{pathlib.Path(__file__).parent!s}/data/qc_nuclei_thresholds_default.yml"
)


def identify_outliers(
    df: pd.DataFrame,
    feature_thresholds: Union[Dict[str, float], str],
    feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
    include_threshold_scores: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """
    This function uses z-scoring to format the data for detecting outlier
    nuclei or cells using specific CellProfiler features. Thresholds are
    the number of standard deviations away from the mean, either above
    (positive) or below (negative). We recommend making sure to not use a
    threshold of 0 as that would represent the whole dataset.

    Args:
        df: pd.DataFrame
            Data frame with converted output from CytoTable.
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

    Returns:
        Union[pd.Series, pd.DataFrame]:
            Outlier series with booleans based on whether outliers were detected
            or not for use within other functions.
    """

    # create a copy of the dataframe to ensure
    # we don't modify the supplied dataframe inplace.
    outlier_df = df.copy()

    thresholds_name = (
        f"outlier_{feature_thresholds}"
        if isinstance(feature_thresholds, str)
        else "outlier_custom"
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
        outlier_df[f"Z_Score_{feature}"] = scipy_zscore(df[feature])
        zscore_columns[feature] = f"Z_Score_{feature}"

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

    return (
        # create a boolean pd.series identifier for dataframe
        # based on all conditions for use within other functions.
        reduce(operator.and_, conditions)
        if not include_threshold_scores
        # otherwise, provide the threshold zscore col and the above column
        else pd.concat(
            [
                # grab only the outlier zscore columns from the outlier_df
                outlier_df[zscore_columns.values()],
                pd.DataFrame({thresholds_name: reduce(operator.and_, conditions)}),
            ],
            axis=1,
        )
    )


def find_outliers(
    df: pd.DataFrame,
    metadata_columns: List[str],
    feature_thresholds: Union[Dict[str, float], str],
    feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
) -> pd.DataFrame:
    """
    This function uses identify_outliers to return a dataframe
    with only the outliers and provided metadata columns.

    Args:
        df: pd.DataFrame
            Data frame with converted output from CytoTable.
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

    Returns:
        pd.DataFrame:
            Outlier data frame for the given conditions.
    """

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
    print("Number of outliers:", outliers_df.shape[0])
    print("Outliers Range:")
    for feature in feature_thresholds:
        print(f"{feature} Min:", outliers_df[feature].min())
        print(f"{feature} Max:", outliers_df[feature].max())

    # Include metadata columns in the output DataFrame
    columns_to_include = list(feature_thresholds.keys()) + metadata_columns

    # Return outliers DataFrame with specified columns
    return outliers_df[columns_to_include]


def label_outliers(
    df: pd.DataFrame,
    feature_thresholds: Optional[Union[Dict[str, float], str]] = None,
    feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
    include_threshold_scores: bool = False,
) -> pd.DataFrame:
    """
    Use identify_outliers to label the original dataset for
    where a cell passed or failed the quality control condition(s).

        Args:
            df: pd.DataFrame
                Data frame with converted output from CytoTable.
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

        Returns:
            pd.DataFrame:
                Full dataframe with optional scores and outlier boolean column.
    """

    # for single outlier processing
    if isinstance(feature_thresholds, (str, dict)):
        # return the outlier dataframe for one threshold rule
        identified_outliers = identify_outliers(
            df=df,
            feature_thresholds=feature_thresholds,
            feature_thresholds_file=feature_thresholds_file,
            include_threshold_scores=include_threshold_scores,
        )
        return pd.concat(
            [
                df,
                (
                    identified_outliers
                    if isinstance(identified_outliers, pd.DataFrame)
                    else pd.DataFrame(
                        {
                            (
                                f"outlier_{feature_thresholds}"
                                if isinstance(feature_thresholds, str)
                                else "outlier_custom"
                            ): identified_outliers
                        }
                    )
                ),
            ],
            axis=1,
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
        return labeled_df.loc[:, ~labeled_df.columns.duplicated()]


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
