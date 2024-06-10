"""
Defines a QCDataFrame class for use in coSMicQC.
"""

from typing import Any, Dict, Self, Union

import pandas as pd


class QCDataFrame:
    """
    A class to handle and load different types of data files into a pandas DataFrame.

    This class can initialize with either a pandas DataFrame or a file path (CSV, TSV,
    TXT, or Parquet). When initialized with a file path, it reads the data into a
    pandas DataFrame.

    Attributes:
        reference (str):
            A string indicating the type of data source, either 'pd.DataFrame'
            or the file path.
        data (pd.DataFrame):
            The loaded data in a pandas DataFrame.

    Methods:
        __call__():
            Returns the underlying pandas DataFrame.
    """

    def __init__(
        self: Self, data: Union[pd.DataFrame, str], **kwargs: Dict[str, Any]
    ) -> None:
        """
        Initializes the QCDataFrame with either a DataFrame or a file path.

        Args:
            data (Union[pd.DataFrame, str]):
                The data source, either a pandas DataFrame or a file path.
            **kwargs:
                Additional keyword arguments to pass to the pandas read functions.
        """
        if isinstance(data, pd.DataFrame):
            # if data is a pd.DataFrame, remember this within the reference attr
            self.reference = "pd.DataFrame"
            self.data = data
        elif isinstance(data, str):
            # if the data is a string, remember the original source
            # through a reference attr
            self.reference = data

            # Read the data from the file based on its extension
            if data.endswith(".csv"):
                self.data = pd.read_csv(data, **kwargs)
            elif data.endswith(".tsv") or data.endswith(".txt"):
                self.data = pd.read_csv(data, delimiter="\t", **kwargs)
            elif data.endswith(".parquet"):
                self.data = pd.read_parquet(data, **kwargs)

    def __call__(self: Self) -> pd.DataFrame:
        """
        Returns the underlying pandas DataFrame.

        Returns:
            pd.DataFrame: The data in a pandas DataFrame.
        """
        return self.data
