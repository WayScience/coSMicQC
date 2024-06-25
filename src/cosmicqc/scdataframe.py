"""
Defines a SCDataFrame class for use in coSMicQC.
"""

import pathlib
from typing import Any, Dict, TypeVar, Union

import pandas as pd

# provide backwards compatibility for Self type in earlier Python versions.
# see: https://peps.python.org/pep-0484/#annotating-instance-and-class-methods
SCDataFrame_type = TypeVar("SCDataFrame_type", bound="SCDataFrame")


class SCDataFrame:
    """
    A class designed to enhance single-cell data handling by wrapping
    pandas DataFrame capabilities, providing advanced methods for quality control,
    comprehensive analysis, and image-based data processing.

    This class can initialize with either a pandas DataFrame or a file path (CSV, TSV,
    TXT, or Parquet). When initialized with a file path, it reads the data into a
    pandas DataFrame. It also includes capabilities to export data.

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
            Returns a representational string of the underlying pandas DataFrame.
        __getattr__():
            Returns the underlying attributes of the pandas DataFrame.
        __getitem__():
            Returns slice of data from pandas DataFrame.
    """

    def __init__(
        self: SCDataFrame_type,
        data: Union[SCDataFrame_type, pd.DataFrame, str, pathlib.Path],
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Initializes the SCDataFrame with either a DataFrame or a file path.

        Args:
            data (Union[pd.DataFrame, str]):
                The data source, either a pandas DataFrame or a file path.
            **kwargs:
                Additional keyword arguments to pass to the pandas read_* methods.
        """

        if isinstance(data, SCDataFrame):
            # if data is an instance of SCDataFrame, use its data_source and data
            self.data_source = data.data_source
            self.data = data.data

        elif isinstance(data, pd.Series):
            # if data is a pd.Series, remember this within the data_source attr
            self.data_source = "pandas.Series"
            # also cast the series to a dataframe
            self.data = pd.DataFrame(data)

        elif isinstance(data, pd.DataFrame):
            # if data is a pd.DataFrame, remember this within the data_source attr
            self.data_source = "pandas.DataFrame"
            self.data = data

        elif isinstance(data, (pathlib.Path, str)):
            # if the data is a string or a pathlib path, remember the original source
            # through a data_source attr
            self.data_source = data

            # interpret the data through pathlib
            data_path = pathlib.Path(data)

            # Read the data from the file based on its extension
            if (
                data_path.suffix == ".csv"
                or data_path.suffix in (".tsv", ".txt")
                or data_path.suffixes == [".csv", ".gz"]
                or data_path.suffixes == [".tsv", ".gz"]
            ):
                # read as a CSV, CSV.GZ, .TSV, or .TXT file
                self.data = pd.read_csv(data, **kwargs)
            elif data_path.suffix == ".parquet":
                # read as a Parquet file
                self.data = pd.read_parquet(data, **kwargs)
            else:
                raise ValueError("Unsupported file format for SCDataFrame.")
        else:
            raise ValueError("Unsupported data type for SCDataFrame.")

    def export(
        self: SCDataFrame_type, file_path: str, **kwargs: Dict[str, Any]
    ) -> None:
        """
        Exports the underlying pandas DataFrame to a file.

        Args:
            file_path (str):
                The path where the DataFrame should be saved.
            **kwargs:
                Additional keyword arguments to pass to the pandas to_* methods.
        """

        data_path = pathlib.Path(file_path)

        # export to csv
        if ".csv" in data_path.suffixes:
            self.data.to_csv(file_path, **kwargs)
        # export to tsv
        elif any(elem in data_path.suffixes for elem in (".tsv", ".txt")):
            self.data.to_csv(file_path, sep="\t", **kwargs)
        # export to parquet
        elif data_path.suffix == ".parquet":
            self.data.to_parquet(file_path, **kwargs)
        else:
            raise ValueError("Unsupported file format for export.")

    def __call__(self: SCDataFrame_type) -> pd.DataFrame:
        """
        Returns the underlying pandas DataFrame.

        Returns:
            pd.DataFrame: The data in a pandas DataFrame.
        """
        return self.data

    def __repr__(self: SCDataFrame_type) -> str:
        """
        Returns the representation of the underlying pandas DataFrame.

        Returns:
            pd.DataFrame: The data in a pandas DataFrame.
        """
        return repr(self.data)

    def __getattr__(self: SCDataFrame_type, attr: str) -> Any:  # noqa: ANN401
        """
        Intercept attribute accesses and delegate them to the underlying
        pandas DataFrame, except for custom methods.

        Args:
            attr (str):
                The name of the attribute being accessed.

        Returns:
            Any:
                The value of the attribute from the pandas DataFrame.
        """
        if attr in self.__dict__:
            return self.__dict__[attr]
        return getattr(self.data, attr)

    def __getitem__(self: SCDataFrame_type, key: Union[int, str]) -> Any:  # noqa: ANN401
        """
        Returns an element or a slice of the underlying pandas DataFrame.

        Args:
            key:
                The key or slice to access the data.

        Returns:
            pd.DataFrame or any:
                The selected element or slice of data.
        """
        return self.data[key]
