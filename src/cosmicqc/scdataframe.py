"""
Defines a SCDataFrame class for use in coSMicQC.
"""

import pathlib
from typing import Any, Dict, TypeVar, Union

import pandas as pd

# provide backwards compatibility for Self type in earlier Python versions.
# see: https://peps.python.org/pep-0484/#annotating-instance-and-class-methods
Self_SCDataFrame = TypeVar("Self_SCDataFrame", bound="SCDataFrame")


class SCDataFrame:
    """
    A class to handle and load different types of data files into a pandas DataFrame.

    This class can initialize with either a pandas DataFrame or a file path (CSV, TSV,
    TXT, or Parquet). When initialized with a file path, it reads the data into a
    pandas DataFrame.

    Attributes:
        data_source (str):
            A string indicating the data source, either 'pd.DataFrame'
            or the file path.
        data (pd.DataFrame):
            The loaded data in a pandas DataFrame.

    Methods:
        __call__():
            Returns the underlying pandas DataFrame.
        __repr__():
            Returns representation of underlying pandas DataFrame.
        __getattr__():
            Returns underlying attributes of pandas DataFrame.
        __getitem__():
            Returns slice of data from pandas DataFrame.
    """

    def __init__(
        self: Self_SCDataFrame, data: Union[pd.DataFrame, str], **kwargs: Dict[str, Any]
    ) -> None:
        """
        Initializes the SCDataFrame with either a DataFrame or a file path.

        Args:
            data (Union[pd.DataFrame, str]):
                The data source, either a pandas DataFrame or a file path.
            **kwargs:
                Additional keyword arguments to pass to the pandas read functions.
        """

        if isinstance(data, pd.DataFrame):
            # if data is a pd.DataFrame, remember this within the data_source attr
            self.data_source = "pd.DataFrame"
            self.data = data

        elif isinstance(data, pathlib.Path | str):
            # if the data is a string, remember the original source
            # through a data_source attr
            self.data_source = data

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
                raise ValueError("Unsupported file format for SCDataFrame.")
        else:
            raise ValueError("Unsupported input type for SCDataFrame.")

    def __call__(self: Self_SCDataFrame) -> pd.DataFrame:
        """
        Returns the underlying pandas DataFrame.

        Returns:
            pd.DataFrame: The data in a pandas DataFrame.
        """
        return self.data

    def __repr__(self: Self_SCDataFrame) -> pd.DataFrame:
        """
        Returns the representation of underlying pandas DataFrame.

        Returns:
            pd.DataFrame: The data in a pandas DataFrame.
        """
        return repr(self.data)

    def __getattr__(self: Self_SCDataFrame, attr: str) -> Any:  # noqa: ANN401
        """
        Intercept attribute accesses and delegate them to the underlying
        pandas DataFrame.

        Args:
            attr (str): The name of the attribute being accessed.

        Returns:
            Any: The value of the attribute from the pandas DataFrame.
        """
        return getattr(self.data, attr)

    def __getitem__(self: Self_SCDataFrame, key: Union[int, str]) -> Any:  # noqa: ANN401
        """
        Returns an element or a slice of the underlying pandas DataFrame.

        Args:
            key: The key or slice to access the data.

        Returns:
            pd.DataFrame or any: The selected element or slice of data.
        """
        return self.data[key]
