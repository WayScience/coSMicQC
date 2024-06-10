"""
Defines a QCDataFrame class for use in coSMicQC.
"""

import pathlib
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

        # print(data)
        # print(type(data))
        # print(isinstance(data, QCDataFrame))

        if isinstance(data, pd.DataFrame):
            # if data is a pd.DataFrame, remember this within the reference attr
            self.reference = "pd.DataFrame"
            self.data = data

        elif isinstance(data, pathlib.Path | str):
            # if the data is a string, remember the original source
            # through a reference attr
            self.reference = data

            # interpret the data through pathlib
            data_path = pathlib.Path(data)

            # Read the data from the file based on its extension
            if data_path.suffix == ".csv":
                # read as a CSV
                self.data = pd.read_csv(data, **kwargs)
            elif data_path.suffix in (".tsv", ".txt"):
                # read as a TSV
                self.data = pd.read_csv(data, delimiter="\t", **kwargs)
            elif data_path.suffix == ".parquet":
                # read as a Parquet file
                self.data = pd.read_parquet(data, **kwargs)

        else:
            raise ValueError("Unsupported file format for QCDataFrame.")

    def __call__(self: Self) -> pd.DataFrame:
        """
        Returns the underlying pandas DataFrame.

        Returns:
            pd.DataFrame: The data in a pandas DataFrame.
        """
        return self.data

    def __repr__(self: Self) -> pd.DataFrame:
        """
        Returns the underlying pandas DataFrame.

        Returns:
            pd.DataFrame: The data in a pandas DataFrame.
        """
        return self.data

    def __getattr__(self: Self, attr: str) -> Any:  # noqa: ANN401
        """
        Intercept attribute accesses and delegate them to the underlying
        pandas DataFrame.

        Args:
            attr (str): The name of the attribute being accessed.

        Returns:
            Any: The value of the attribute from the pandas DataFrame.
        """
        return getattr(self.data, attr)

    def __getitem__(self: Self, key: Union[int, str]) -> Any:  # noqa: ANN401
        """
        Returns an element or a slice of the underlying pandas DataFrame.

        Args:
            key: The key or slice to access the data.

        Returns:
            pd.DataFrame or any: The selected element or slice of data.
        """
        return self.data[key]
