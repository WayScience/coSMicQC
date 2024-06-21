"""
Defines a SCDataFrame class for use in coSMicQC.
"""

import pathlib
import random
import webbrowser
from typing import Any, Dict, List, Optional, TypeVar, Union

import pandas as pd
import plotly
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from IPython import get_ipython
from jinja2 import Environment, FileSystemLoader

# provide backwards compatibility for Self type in earlier Python versions.
# see: https://peps.python.org/pep-0484/#annotating-instance-and-class-methods
SCDataFrame_type = TypeVar("SCDataFrame_type", bound="SCDataFrame")


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
                Additional keyword arguments to pass to the pandas read functions.
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

        elif isinstance(data, pd.Series):
            # if data is a pd.DataFrame, remember this within the data_source attr
            self.data_source = "pandas.Series"
            self.data = pd.DataFrame(data)

        elif isinstance(data, (pathlib.Path, str)):
            # if the data is a string, remember the original source
            # through a data_source attr
            self.data_source = data

            # interpret the data through pathlib
            data_path = pathlib.Path(data)

            # Read the data from the file based on its extension
            if (
                data_path.suffix == ".csv"
                or data_path.suffix in (".tsv", ".txt")
                or data_path.suffixes == [".csv", ".gz"]
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
            file_path (str): The path where the DataFrame should be saved.
            **kwargs: Additional keyword arguments to pass to the pandas to_* methods.
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

    @staticmethod
    def is_notebook_or_lab() -> bool:
        """
        Determines if the code is being executed in a Jupyter notebook or
        JupyterLab environment.

        This method attempts to detect the interactive shell environment
        using IPython's `get_ipython` function. It checks the class name of the current
          IPython shell to distinguish between different execution environments.

        Returns:
            bool:
                - `True`
                    if the code is being executed in a Jupyter notebook or JupyterLab.
                - `False`
                    otherwise (e.g., standard Python shell, terminal IPython shell,
                    or scripts).
        """
        try:
            # check for type of session via ipython
            shell = get_ipython().__class__.__name__
            if "ZMQInteractiveShell" in shell:
                return True
            elif "TerminalInteractiveShell" in shell:
                return False
            else:
                return False
        except NameError:
            return False

    def show_report(
        self: SCDataFrame_type,
        report_path: Optional[str] = None,
        auto_open: bool = True,
        color_palette: Optional[List[str]] = None,
    ) -> None:
        """
        Generates and displays a report based on the current DataFrame's data
        quality control (DQC) columns.

        This method organizes the DQC columns from the DataFrame, creates
        visualizations for each threshold set, and then either displays the
        visualizations inline (if running in a Jupyter notebook or lab) or
        opens an HTML report in the default web browser.

        Args:
            report_path (Optional[str]):
                The file path where the HTML report should be saved and displayed.
                If `None`, the report will be displayed inline if in a notebook
                or lab environment.
            auto_open: bool:
                Whether to automatically open the report.
            color_palette Optional(List[str]):
                Optional list for color palette to use.

        Raises:
            ValueError: If the DataFrame does not contain any DQC columns.
        """

        # find all cosmicqc columns in the data using the prefix `cqc.`
        cqc_cols = [col for col in self.data.columns.tolist() if "cqc." in col]
        # organize column data into the threshold set name, threshold is_outlier col,
        # and the threshold score columns as list
        organized_columns = [
            [
                # name of the threshold set
                threshold_set,
                # column which includes boolean is_outlier data for threshold set
                next(
                    (
                        col
                        for col in cqc_cols
                        if f"cqc.{threshold_set}.is_outlier" in col
                    ),
                    None,
                ),
                # columns which show the data associated with thresholds
                [col for col in cqc_cols if f"cqc.{threshold_set}.Z_Score." in col],
            ]
            for threshold_set in sorted({col.split(".")[1] for col in cqc_cols})
        ]

        # create figures for visualization based on the name, outlier status,
        # and threshold scores
        figures = [
            self.create_threshold_set_outlier_visualization(
                df=self.data,
                threshold_set_name=set_name,
                col_outlier=col_outlier,
                cols_threshold_scores=cols_threshold_scores,
                color_palette=color_palette,
            )
            for set_name, col_outlier, cols_threshold_scores in organized_columns
        ]

        # if we're running in a notebook or jupyter lab, show the figures as-is
        if self.is_notebook_or_lab() or report_path is None:
            # if we should automatically open, show the figures
            if auto_open:
                for figure in figures:
                    figure.show()

            return figures

        # otherwise, create an html file with figures and open it with default browser
        else:
            html_path = self.create_figure_group_html(
                figures=figures, report_path=report_path
            )

            # if we should auto open, show the html file in default web browser
            if auto_open:
                webbrowser.open(f"file://{pathlib.Path(html_path).resolve()}")

            print(f"Opened default web browser for report {html_path}")

            return html_path

    @staticmethod
    def create_figure_group_html(
        figures: List[plotly.graph_objs._figure.Figure],
        report_path: Optional[str] = None,
    ) -> str:
        """
        Generates an HTML file containing multiple Plotly figures.

        This method takes a list of Plotly figure objects, converts them to HTML,
        and embeds them into a template HTML file. The resulting HTML file is then
        saved to the specified path.

        Args:
            figures (List[plotly.graph_objs._figure.Figure]):
                A list of Plotly figure objects to be included in the HTML report.
            report_path (str):
                The file path where the HTML report will be saved.
                Defaults to "cosmicqc_outlier_report.html" when None.

        Returns:
            str: The path to the saved HTML report.
        """

        # if we have none for the report path, use a default name.
        if report_path is None:
            report_path = "cosmicqc_outlier_report.html"

        # create wrapped html for figures
        figure_html = "".join(
            [
                f"<div class='fig_wrapper'>{fig.to_html(full_html=False)}</div>"
                for fig in figures
            ]
        )

        # configure jinja environment
        env = Environment(
            loader=FileSystemLoader(f"{pathlib.Path(__file__).parent!s}/data")
        )
        # load a jinja template
        template = env.get_template("report_template.html")

        # Render the template with Plotly figure HTML
        rendered_html = template.render(figure_html=figure_html)

        # write the html to file
        with open(report_path, "w") as f:
            f.write(rendered_html)

        # return the path of the file
        return report_path

    def create_threshold_set_outlier_visualization(  # noqa: PLR0913
        self: SCDataFrame_type,
        df: pd.DataFrame,
        threshold_set_name: str,
        col_outlier: str,
        cols_threshold_scores: List[str],
        color_palette: Optional[List[str]] = None,
    ) -> plotly.graph_objs._figure.Figure:
        """
        Creates a Plotly figure visualizing the Z-score distributions and outliers
        for a given threshold set.

        This method generates histograms for each Z-score column in the given DataFrame,
        colors them based on outlier status, and overlays them into a single figure.

        Args:
            df (pd.DataFrame):
                The DataFrame containing the data to be visualized.
            threshold_set_name (str):
                The name of the threshold set being visualized.
            col_outlier (str):
                The column name indicating outlier status.
            cols_threshold_scores (List[str]):
                A list of column names representing the Z-scores to be visualized.
            color_palette Optional(List[str]):
                Optional list for color palette to use.

        Returns:
            plotly.graph_objs._figure.Figure:
                A Plotly figure object containing the visualization.
        """

        if color_palette is None:
            # Create a list of colors from a Plotly color palette
            color_palette = pc.qualitative.Dark24

        # Create histograms using plotly.express with pattern_shape and random color
        figures = []
        for col in cols_threshold_scores:
            fig = px.histogram(
                df,
                x=col,
                color=col_outlier,
                nbins=50,
                pattern_shape=col_outlier,
                opacity=0.7,
            )
            figures.append(fig)

        # Create a combined figure
        fig = go.Figure()

        # Add traces from each histogram and modify colors, names, and pattern shapes
        for idx, fig_hist in enumerate(figures):
            fig_color = random.choice(color_palette)

            for trace in fig_hist.data:
                trace.marker.color = fig_color
                trace.marker.pattern.shape = (
                    "x" if trace.name == "True" else ""
                )  # Use pattern shapes
                renamed_col = cols_threshold_scores[idx].replace(
                    f"cqc.{threshold_set_name}.Z_Score.", ""
                )
                trace.name = (
                    f"{renamed_col} ({'outlier' if trace.name == 'True' else 'inlier'})"
                )
                # Update hovertemplate to match the name in the key
                trace.hovertemplate = (
                    f"<b>{renamed_col}</b><br>"
                    + "Z-Score: %{x}<br>"
                    + "Single-cell Count (log): %{y}<br>"
                    + "<extra></extra>"
                )
                fig.add_trace(trace)

        # Update layout
        fig.update_layout(
            title=f"{threshold_set_name.replace('_', ' ').title()} Z-Score Outliers",
            xaxis_title="Z-Score",
            yaxis_title="Single-cell Count (log)",
            yaxis_type="log",
            # ensures that histograms are overlapping
            barmode="overlay",
            legend_title_text="Measurement Type and QC Status",
            legend={
                "orientation": "v",
                "yanchor": "top",
                "y": 0.95,
                "xanchor": "left",
                "x": 1.02,
            },
        )

        return fig

    def __call__(self: SCDataFrame_type) -> pd.DataFrame:
        """
        Returns the underlying pandas DataFrame.

        Returns:
            pd.DataFrame: The data in a pandas DataFrame.
        """
        return self.data

    def __repr__(self: SCDataFrame_type) -> pd.DataFrame:
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
            attr (str): The name of the attribute being accessed.

        Returns:
            Any: The value of the attribute from the pandas DataFrame.
        """
        if attr in self.__dict__:
            return self.__dict__[attr]
        return getattr(self.data, attr)

    def __getitem__(self: SCDataFrame_type, key: Union[int, str]) -> Any:  # noqa: ANN401
        """
        Returns an element or a slice of the underlying pandas DataFrame.

        Args:
            key: The key or slice to access the data.

        Returns:
            pd.DataFrame or any: The selected element or slice of data.
        """
        return self.data[key]
