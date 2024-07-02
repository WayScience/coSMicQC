"""
Defines a SCDataFrame class for use in coSMicQC.
"""

import base64
import pathlib
import random
import re
import webbrowser
from io import BytesIO
from pandas._config import (
    get_option,
)
from pandas.io.formats import (
    format as fmt,
)
from io import StringIO
from typing import Any, Dict, List, Optional, TypeVar, Union, Tuple
from functools import partial
import pandas as pd
import plotly
from IPython.display import HTML, display
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import skimage
from IPython import get_ipython
from jinja2 import Environment, FileSystemLoader
from PIL import Image

# provide backwards compatibility for Self type in earlier Python versions.
# see: https://peps.python.org/pep-0484/#annotating-instance-and-class-methods
SCDataFrame_type = TypeVar("SCDataFrame_type", bound="SCDataFrame")


class SCDataFrame(pd.DataFrame):
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

    _metadata = ["_custom_attrs"]

    def __init__(
        self,
        data: Union[pd.DataFrame, str, pathlib.Path],
        data_context_dir: Optional[str] = None,
        data_bounding_box: Optional[pd.DataFrame] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Initializes the SCDataFrame with either a DataFrame or a file path.

        Args:
            data (Union[pd.DataFrame, str, pathlib.Path]):
                The data source, either a pandas DataFrame or a file path.
            data_context_dir (Optional[str]): Directory context for the data.
            data_bounding_box (Optional[pd.DataFrame]): Bounding box data for the DataFrame.
            **kwargs: Additional keyword arguments to pass to the pandas read functions.
        """

        self._custom_attrs = {
            "data_source": None,
            "data_context_dir": None,
            "data_bounding_box": None,
        }

        if data_context_dir is not None:
            self._custom_attrs["data_context_dir"] = data_context_dir

        if isinstance(data, SCDataFrame):
            self._custom_attrs["data_source"] = data._custom_attrs["data_source"]
            self._custom_attrs["data_context_dir"] = data._custom_attrs["data_context_dir"]
            super().__init__(data)
        elif isinstance(data, pd.DataFrame):
            self._custom_attrs["data_source"] = "pandas.DataFrame"
            super().__init__(data)
        elif isinstance(data, pd.Series):
            self._custom_attrs["data_source"] = "pandas.Series"
            super().__init__(pd.DataFrame(data))
        elif isinstance(data, (str, pathlib.Path)):
            data_path = pathlib.Path(data)
            self._custom_attrs["data_source"] = str(data_path)

            if data_context_dir is None:
                self._custom_attrs["data_context_dir"] = str(data_path.parent)
            else:
                self._custom_attrs["data_context_dir"] = data_context_dir

            if data_path.suffix in {".csv", ".tsv", ".txt"} or data_path.suffixes == [
                ".csv",
                ".gz",
            ]:
                data = pd.read_csv(data_path, **kwargs)
            elif data_path.suffix == ".parquet":
                data = pd.read_parquet(data_path, **kwargs)
            else:
                raise ValueError("Unsupported file format for SCDataFrame.")

            super().__init__(data)

        else:
            super().__init__(data)

        if data_bounding_box is None:
            self._custom_attrs["data_bounding_box"] = self.get_bounding_box_from_data()

        else:
            self._custom_attrs["data_bounding_box"] = data_bounding_box

    def __getitem__(
        self: SCDataFrame_type, key: Union[int, str]
    ) -> Any:  # noqa: ANN401
        """
        Returns an element or a slice of the underlying pandas DataFrame.

        Args:
            key:
                The key or slice to access the data.

        Returns:
            pd.DataFrame or any:
                The selected element or slice of data.
        """

        result = super().__getitem__(key)

        if isinstance(result, pd.Series):
            return result

        elif isinstance(result, pd.DataFrame):
            return SCDataFrame(
                super().__getitem__(key),
                data_context_dir=self._custom_attrs["data_context_dir"],
                data_bounding_box=self._custom_attrs["data_bounding_box"],
            )
    
    def _wrap_method(self, method, *args, **kwargs):
        result = method(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            result = SCDataFrame(
                result,
                data_context_dir=self._custom_attrs["data_context_dir"],
                data_bounding_box=self._custom_attrs["data_bounding_box"],
            )
        return result
    
    def sort_values(self, *args, **kwargs):
        return self._wrap_method(super().sort_values, *args, **kwargs)

    def get_bounding_box_from_data(self: SCDataFrame_type):
        # Define column groups and their corresponding conditions
        column_groups = {
            "cyto": [
                "Cytoplasm_AreaShape_BoundingBoxMaximum_X",
                "Cytoplasm_AreaShape_BoundingBoxMaximum_Y",
                "Cytoplasm_AreaShape_BoundingBoxMinimum_X",
                "Cytoplasm_AreaShape_BoundingBoxMinimum_Y",
            ],
            "nuclei": [
                "Nuclei_AreaShape_BoundingBoxMaximum_X",
                "Nuclei_AreaShape_BoundingBoxMaximum_Y",
                "Nuclei_AreaShape_BoundingBoxMinimum_X",
                "Nuclei_AreaShape_BoundingBoxMinimum_Y",
            ],
            "cells": [
                "Cells_AreaShape_BoundingBoxMaximum_X",
                "Cells_AreaShape_BoundingBoxMaximum_Y",
                "Cells_AreaShape_BoundingBoxMinimum_X",
                "Cells_AreaShape_BoundingBoxMinimum_Y",
            ],
        }

        # Determine which group of columns to select based on availability in self.data
        selected_group = None
        for group, cols in column_groups.items():
            if all(col in self.columns.tolist() for col in cols):
                selected_group = group
                break

        # Assign the selected columns to self.bounding_box_df
        if selected_group:
            return self.filter(items=column_groups[selected_group])

        return None

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
            self.to_csv(file_path, **kwargs)
        # export to tsv
        elif any(elem in data_path.suffixes for elem in (".tsv", ".txt")):
            self.to_csv(file_path, sep="\t", **kwargs)
        # export to parquet
        elif data_path.suffix == ".parquet":
            self.to_parquet(file_path, **kwargs)
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
        cqc_cols = [col for col in self.columns.tolist() if "cqc." in col]
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
                df=self,
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

    def find_image_columns(self: SCDataFrame_type) -> bool:
        pattern = r".*\.(tif|tiff)$"
        return [
            column
            for column in self.columns
            if self[column]
            .apply(
                lambda value: isinstance(value, str)
                and re.match(pattern, value, flags=re.IGNORECASE)
            )
            .any()
        ]

    @staticmethod
    def process_image_data_as_html_display(
        data_value: Any,
        data_context_dir: str = None,
        bounding_box=Tuple[int, int, int, int],
    ) -> str:

        if not pathlib.Path(data_value).is_file():
            if not pathlib.Path(
                candidate_path := (f"{data_context_dir}/{data_value}")
            ).is_file():
                return data_value
            else:
                data_value = candidate_path

        # Read the TIFF image from the byte array
        tiff_image = skimage.io.imread(data_value)

        # Convert the image array to a PIL Image
        pil_image = Image.fromarray(tiff_image)

        cropped_img = pil_image.crop(bounding_box)

        # Save the PIL Image as PNG to a BytesIO object
        png_bytes_io = BytesIO()
        cropped_img.save(png_bytes_io, format="PNG")

        # Get the PNG byte data
        png_bytes = png_bytes_io.getvalue()

        return f'<img src="data:image/png;base64,{base64.b64encode(png_bytes).decode("utf-8")}" style="width:300px;"/>'

    def get_displayed_rows(self: SCDataFrame_type) -> List[int]:
        # Get the current display settings
        max_rows = pd.get_option("display.max_rows")
        min_rows = pd.get_option("display.min_rows")

        if len(self) <= max_rows:
            # If the DataFrame has fewer rows than max_rows, all rows will be displayed
            return self.index.tolist()
        else:
            # Calculate how many rows will be displayed at the beginning and end
            half_min_rows = min_rows // 2
            start_display = self.index[:half_min_rows].tolist()
            end_display = self.index[-half_min_rows:].tolist()
            return start_display + end_display

    def _repr_html_(
        self: SCDataFrame_type, key: Optional[Union[int, str]] = None
    ) -> str:
        """
        Returns HTML representation of the underlying pandas DataFrame
        for use within Juypyter notebook environments and similar.

        Referenced with modifications from:
        https://github.com/pandas-dev/pandas/blob/v2.2.2/pandas/core/frame.py#L1216

        Return a html representation for a particular DataFrame.

        Mainly for Jupyter notebooks.

        Returns:
            str: The data in a pandas DataFrame.
        """

        if self._info_repr():
            buf = StringIO()
            self.info(buf=buf)
            # need to escape the <class>, should be the first line.
            val = buf.getvalue().replace("<", r"&lt;", 1)
            val = val.replace(">", r"&gt;", 1)
            return f"<pre>{val}</pre>"

        if get_option("display.notebook_repr_html"):
            max_rows = get_option("display.max_rows")
            min_rows = get_option("display.min_rows")
            max_cols = get_option("display.max_columns")
            show_dimensions = get_option("display.show_dimensions")

            # determine if we have image_cols to display
        if image_cols := self.find_image_columns():

            # readd bounding box cols if they are no longer available as in cases
            # of masking or accessing various pandas attr's
            bounding_box_externally_joined = False

            if self._custom_attrs["data_bounding_box"] is not None and not all(
                col in self.columns.tolist()
                for col in self._custom_attrs["data_bounding_box"].columns.tolist()
            ):

                data = self.join(other=self._custom_attrs["data_bounding_box"])
                bounding_box_externally_joined = True
            else:

                data = self.copy()

            # gather indices which will be displayed based on pandas configuration
            display_indices = self.get_displayed_rows()

            for image_col in image_cols:
                data.loc[display_indices, image_col] = data.loc[display_indices].apply(
                    lambda row: self.process_image_data_as_html_display(
                        data_value=row[image_col],
                        data_context_dir=self._custom_attrs["data_context_dir"],
                        bounding_box=(
                            row["Cytoplasm_AreaShape_BoundingBoxMinimum_X"],
                            row["Cytoplasm_AreaShape_BoundingBoxMinimum_Y"],
                            row["Cytoplasm_AreaShape_BoundingBoxMaximum_X"],
                            row["Cytoplasm_AreaShape_BoundingBoxMaximum_Y"],
                        ),
                    ),
                    axis=1,
                )

            if bounding_box_externally_joined:
                data = data.drop(
                    self._custom_attrs["data_bounding_box"].columns.tolist(), axis=1
                )

            formatter = fmt.DataFrameFormatter(
                data,
                columns=None,
                col_space=None,
                na_rep="NaN",
                formatters=None,
                float_format=None,
                sparsify=None,
                justify=None,
                index_names=True,
                header=True,
                index=True,
                bold_rows=True,
                # note: we avoid escapes to allow HTML rendering for images
                escape=False,
                max_rows=max_rows,
                min_rows=min_rows,
                max_cols=max_cols,
                show_dimensions=show_dimensions,
                decimal=".",
            )

            return fmt.DataFrameRenderer(formatter).to_html()

        else:
            return None
