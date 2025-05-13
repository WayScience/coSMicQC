"""
Module for detecting contamination (e.g., mycoplasma) from the morphology profiles.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cosmicqc import find_outliers


class ContaminationDetector:
    """_
    This class implements a contamination detection process for high-content morphology data.
    There are three steps to the process:

    1. Check for skewness in the cytoplasm texture and variability in the FormFactor features.
    If either is detected, the process moves on to the next step.
    2. Determine if the whole plate or partial plate is impacted by checking the mean values of the features.
    The process will move on to the last step only if partial contamination is detected in the texture feature.
    3. If partial contamination is detected for texture, find outliers and plot the proportion of outliers per well.

    Attributes
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing the features to be analyzed.
        Should be the output from the CytoTable convert function.
    nucleus_channel_naming : str
        The naming convention for the nucleus channel in the DataFrame.
    cyto_feature : str
        The feature name for cytoplasm texture.
    formfactor_feature : str
        The feature name for FormFactor.
    is_skewed : bool
        Indicates if the distribution is skewed.
    is_variable : bool
        Indicates if the distribution is variable.
    partial_contamination_texture_detected : bool
        Indicates if partial contamination is detected in the texture feature.

    Methods
    -------
    skewness_test_cytoplasm_texture()
        Calculate Bowley's skewness for the cytoplasm texture feature and set thresholds
        for detecting abnormal texture around the nucleus.
    variability_test_formfactor()
        Calculate the interquartile range (IQR) for the FormFactor feature to determine
        abnormal variability of nucleus shape.
    step_one_test()
        Run the first step of the contamination detection process by detecting either
        texture skew and/or FormFactor variability.
    check_texture_mean()
        Check the mean value of the cytoplasm texture feature to determine if whole
        plate contamination is present.
    check_formfactor_mean()
        Check the mean value of the FormFactor feature to determine if whole plate
        contamination is present.
    step_two_test()
        Run the second step of the contamination detection process to determine if whole
        plate or partial plate contamination is present based on the mean of the feature.
    find_texture_outliers()
        Find outliers in the texture feature using the coSMicQC find_outliers function.
    plot_proportion_outliers()
        Plot the proportion of outliers detected in each well.
    step_three_test()
        Run the third step of the contamination detection process to find outliers and
        plot the proportion of outliers per well.
    run_all_tests()
        Run all three steps of the contamination detection process with conditional logic.
    """  # noqa: E501

    def __init__(
        self, dataframe: pd.DataFrame, nucleus_channel_naming: str = "DNA"
    ) -> None:
        """
        __init__ function for the ContaminationDetector class.

        Args:
            dataframe (pd.DataFrame):
                The input DataFrame containing the features to be analyzed.
                Should be the output from the CytoTable convert function.
            nucleus_channel_naming (str, optional):
                The naming convention for the nucleus channel in the DataFrame.
                Defaults to "DNA".
        Raises:
            ValueError: Raised if setting either report path or name without the other.
        """
        self.dataframe = dataframe
        self.nucleus_channel_naming = nucleus_channel_naming

        # set the features to be used for contamination detection
        self.cyto_feature = (
            f"Cytoplasm_Texture_InfoMeas1_{self.nucleus_channel_naming}_3_02_256"
        )
        self.formfactor_feature = "Nuclei_AreaShape_FormFactor"

    def skewness_test_cytoplasm_texture(self) -> bool:
        """
        Bowley's skewness score is calculated for the cytoplasm texture
        around the nucleus feature.
        Thresholds are assigned to determine if there is abnormal texture around
        the nucleus, either based on a serious negative skew
        (indicating a whole plate issue) or a positive skew
        (indicating a partial plate issue).
        Thresholds are set based on findings from multiple experiments.

        Returns:
            boolean: True if the distribution is skewed, False otherwise.
        """
        # calculate the quartiles
        q1 = np.percentile(self.dataframe[self.cyto_feature].dropna(), 25)
        q2 = np.percentile(self.dataframe[self.cyto_feature].dropna(), 50)
        q3 = np.percentile(self.dataframe[self.cyto_feature].dropna(), 75)

        # calculate Bowley's skewness (to detect partial contamination)
        bowley_skewness = (q3 + q1 - 2 * q2) / (q3 - q1)

        # Determine if the distribution is skewed as boolean
        is_skewed = False

        # Check for serious negative skew (might indicate whole plate issue)
        if bowley_skewness < -0.15 or bowley_skewness > 0.09:  # noqa: PLR2004
            is_skewed = True

        return is_skewed

    def variability_test_formfactor(self) -> bool:
        """
        Calculate IQR score for the FormFactor feature to determine if there is
        abnormal variability in the shape of the nuclei.
        More variability in this feature indicates a higher proportion
        of non-circular nuclei.
        This is a good indicator of segmentation issues or contamination.
        Threshold is set based on findings from multiple experiments.

        Returns:
            boolean: True if the distribution is variable, False otherwise.
        """
        # use IQR to measure variability of the feature
        vals = self.dataframe[self.formfactor_feature].dropna()
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        # determine if distribution is highly variable
        is_variable = iqr > 0.15  # noqa: PLR2004

        return is_variable

    def step_one_test(self) -> None:
        """
        Step 1: Check for skewness in the cytoplasm texture around the nucleus and
        variability in the FormFactor of the nucleus.

        Interpretations:
        - If skewed and variable
            → likely contamination with segmentation impact. Move to step 2.
        - If skewed only
            → likely contamination without segmentation impact. Move to step 2.
        - If variable only
            → possible contamination or segmentation issue/phenotype. move to step 2.
        - If neither
            → no strong indication of contamination or segmentation issues. Stop here.
        """
        print("Running step 1...")

        # perform skewness and variability tests
        is_skewed = self.skewness_test_cytoplasm_texture()
        is_variable = self.variability_test_formfactor()

        # set boolean status to use in next steps downstream
        self.is_skewed = is_skewed
        self.is_variable = is_variable

        # set interpretations based on results
        if is_skewed and is_variable:
            interpretation = (
                "Contamination detected! 🚨\n"
                "Anomalous texture around nuclei detected and "
                "nuclei segmentation is impacted.\n"
                "Proceeding to step 2..."
            )
        elif is_skewed:
            interpretation = (
                "Contamination detected! 🚨\n"
                "Anomalous texture around nuclei detected but "
                "nuclei segmentation not clearly impacted.\n"
                "Proceeding to step 2..."
            )
        elif is_variable:
            interpretation = (
                "Potential contamination detected! 🛑\n"
                "This could indicate contamination, segmentation issues, "
                "or an interesting phenotype.\n"
                "Proceeding to step 2..."
            )
        else:
            interpretation = "No indication of contamination. Plate appears clean 🫧!"

        print(
            f"Is the distribution skewed/anomalous texture around the nucleus?: {is_skewed}"  # noqa: E501
        )
        print(
            f"Are the values for nuclei shape variable/unexpected amount of non-circular nuclei? {is_variable}"  # noqa: E501
        )
        print(interpretation)

    def check_texture_mean(self) -> bool:
        """
        Check the mean value of the cytoplasm texture around the nucleus feature to
        determine if whole plate or partial plate contamination is present.
        Threshold is set based on findings from multiple experiments.

        Returns:
            boolean: True if whole plate contamination is detected, False otherwise
            indicates partial contamination.
        """
        # calculate the mean
        mean_value = self.dataframe[self.cyto_feature].mean()
        whole_plate_contamination = mean_value >= -0.25  # noqa: PLR2004

        return whole_plate_contamination  # True = whole plate, False = partial plate

    def check_formfactor_mean(self) -> bool:
        """
        Check the mean value of the FormFactor of the nucleus to determine
        if whole plate or partial plate contamination is present.
        Threshold is set based on findings from multiple experiments.

        Returns:
            boolean: True if whole plate contamination is detected, False otherwise
            indicates partial contamination.
        """
        # calculate the mean
        mean_value = self.dataframe[self.formfactor_feature].mean()
        whole_plate_contamination = mean_value <= 0.78  # noqa: PLR2004

        return whole_plate_contamination  # True = whole plate, False = partial plate

    def step_two_test(self) -> None:
        """
        Step 2: If skewness and/or variability detected, determine if the whole plate or
        partial plate is impacted using the mean values of the features.

        Interpretations:
        - If whole plate contamination detected in both texture and formfactor
            → major warning.
        - If whole plate contamination detected in texture only
            → check nucleus channel images for contamination.
        - If whole plate contamination detected in formfactor only
            → check segmentation parameters and/or images for contamination.
        - If partial plate contamination detected in texture only
            → proceed to step 3.
        - If partial plate contamination detected in formfactor only
            → recommend performing 'find_outliers' with FormFactor.
        """
        # run sanity check to ensure step one was ran prior to step two
        if not hasattr(self, "is_skewed") or not hasattr(self, "is_variable"):
            raise RuntimeError(
                "You must run the first test before performing the second test."
            )

        # catch case where steps are used individually and no contamination is present
        if not self.is_skewed and not self.is_variable:
            print("Data is neither skewed nor variable based on step one.")
            print(
                "Interpretation: No indication of contamination. Plate appears clean 🫧."
            )
            self.partial_contamination_texture_detected = False
            return

        print("Running step 2...")

        # Instantiate variables to check for whole plate contamination
        self.whole_plate_contamination_texture = False
        self.whole_plate_contamination_formfactor = False

        # Check for texture contamination
        if self.is_skewed:
            self.whole_plate_contamination_texture = self.check_texture_mean()
            print(
                f"Whole Plate Contamination Texture: {self.whole_plate_contamination_texture}"  # noqa: E501
            )

        # Check for formfactor contamination
        if self.is_variable:
            self.whole_plate_contamination_formfactor = self.check_formfactor_mean()
            print(
                f"Whole Plate Contamination FormFactor: {self.whole_plate_contamination_formfactor}"  # noqa: E501
            )

        # Determine interpretation
        if (
            self.whole_plate_contamination_texture
            and self.whole_plate_contamination_formfactor
        ):
            interpretation = (
                "MAJOR WARNING! 💥\n"
                "Contamination across entire plate detected in both texture and nuclei "
                "shape features.\n"
                "Strongly suggest inspecting the nucleus channel images for"
                "contamination."
            )
        elif self.whole_plate_contamination_texture:
            interpretation = (
                "The whole plate is contaminated based on cytoplasm texture in nucleus "
                "channel.\n"
                "Please check your nucleus channel images for contamination by "
                "brightening the random FOVs."
            )
        elif self.whole_plate_contamination_formfactor:
            interpretation = (
                "The whole plate is contaminated or the segmentation parameters are "
                "non-optimal based on the shape of the nuclei.\n"
                "Please check your segmentation parameters and/or the images for "
                "contamination by brightening the images."
            )
        elif (
            self.is_skewed
            and not self.whole_plate_contamination_texture
            and not self.is_variable
        ):
            interpretation = (
                "Partial plate contamination detected based on cytoplasm texture "
                "in nucleus channel.\n"
                "Proceeding to step 3..."
            )
        elif (
            self.is_variable
            and not self.whole_plate_contamination_formfactor
            and not self.is_skewed
        ):
            interpretation = (
                "Partial shape deviation detected — possibly an interesting phenotype "
                "or poor segmentation.\n"
                "Recommend running 'find_outliers' with FormFactor to "
                "further evaluate segmentations."
            )

        # Set partial contamination flag for texture to True if meets conditions
        # (used in step 3)
        self.partial_contamination_texture_detected = (
            self.is_skewed and not self.whole_plate_contamination_texture
        )

        print(interpretation)

    def find_texture_outliers(self) -> pd.DataFrame:
        """
        Use coSMicQC find_outliers function to identify contaminated single-cells based
        on texture in the cytoplasm around the nucleus in partially contaminated plates.

        Returns:
            pd.DataFrame: DataFrame containing the outliers detected to use to plotting
            and calculating proportions.
        """
        # necessary metadata columns for processing
        metadata_columns = [
            "Image_Metadata_Plate",
            "Image_Metadata_Well",
            "Image_Metadata_Site",
            "Metadata_Nuclei_Location_Center_X",
            "Metadata_Nuclei_Location_Center_Y",
        ]

        # detect outliers based on original texture feature
        texture_feature_thresholds = {
            # outlier threshold for only cytoplasm texture in nuclei channel
            "Cytoplasm_Texture_InfoMeas1_DAPI_3_02_256": 1,
        }

        texture_nuclei_outliers = find_outliers(
            df=self.dataframe,
            metadata_columns=metadata_columns,
            feature_thresholds=texture_feature_thresholds,
        )

        # for added support, also check for outliers using granularity of the cytoplasm
        # in the nuclei channel
        granularity_feature_thresholds = {
            # outlier threshold for only cytoplasm granularity in nuclei channel
            "Cytoplasm_Granularity_2_DAPI": 1,
        }

        granularity_nuclei_outliers = find_outliers(
            df=self.dataframe,
            metadata_columns=metadata_columns,
            feature_thresholds=granularity_feature_thresholds,
        )

        # combine the two outlier dataframes
        combined_outliers = pd.concat(
            [texture_nuclei_outliers, granularity_nuclei_outliers]
        ).drop_duplicates(
            subset=[
                "Image_Metadata_Well",
                "Image_Metadata_Site",
                "Metadata_Nuclei_Location_Center_X",
                "Metadata_Nuclei_Location_Center_Y",
            ],
            keep="first",
        )

        # print total number of outliers detected
        print(f"Total number of outliers detected: {len(combined_outliers)}")

        # return the outliers dataframe
        return combined_outliers

    def plot_proportion_outliers(self) -> pd.DataFrame:
        """
        Plot the proportion of outliers detected in each well.

        Returns:
            pd.DataFrame: DataFrame containing the proportion of outliers per well.
        """
        # check if the contamination status is set
        if not hasattr(self, "partial_contamination_texture_detected"):
            raise RuntimeError("You must run the second test before plotting outliers.")

        # find outliers
        combined_outliers = self.find_texture_outliers()

        # calculate the proportion of outliers per well
        outlier_proportion_per_well = (
            combined_outliers.groupby("Image_Metadata_Well")
            .size()
            .div(self.dataframe.groupby("Image_Metadata_Well").size(), fill_value=0)
            .fillna(0)
        )

        # convert to a DataFrame for easier handling
        outlier_proportion_per_well = outlier_proportion_per_well.reset_index()
        outlier_proportion_per_well.columns = ["Well", "Proportion"]
        outlier_proportion_per_well["Proportion"] *= 100

        # Fix the order of wells
        outlier_proportion_per_well["Well"] = pd.Categorical(
            outlier_proportion_per_well["Well"],
            sorted(
                outlier_proportion_per_well["Well"], key=lambda x: (x[0], int(x[1:]))
            ),
        )
        outlier_proportion_per_well = outlier_proportion_per_well.sort_values("Well")

        # get total cell count per well
        cell_count_per_well = (
            self.dataframe.groupby("Image_Metadata_Well")
            .size()
            .reset_index(name="CellCount")
        )

        # merge into the outlier proportion dataframe
        outlier_proportion_per_well = outlier_proportion_per_well.merge(
            cell_count_per_well, left_on="Well", right_on="Image_Metadata_Well"
        )

        # normalize cell count to [0,1] for colormap mapping
        norm = plt.Normalize(
            outlier_proportion_per_well["CellCount"].min(),
            outlier_proportion_per_well["CellCount"].max(),
        )
        colors = plt.cm.viridis(norm(outlier_proportion_per_well["CellCount"]))

        # plot the proportions
        plt.figure(figsize=(20, 6))
        ax = sns.barplot(
            data=outlier_proportion_per_well,
            x="Well",
            y="Proportion",
            hue="Well",
            palette=list(colors),
            legend=False,
        )
        plt.xlabel("Well")
        plt.ylabel("Proportion")
        plt.xticks(rotation=45)
        plt.ylim(0, 100)

        # Fix: assign colorbar to the axis we just plotted
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])  # this avoids a warning
        plt.colorbar(sm, ax=ax, label="Total Cell Count")

        plt.show()

        return outlier_proportion_per_well

    def step_three_test(self) -> None:
        """
        Step 3: If partial contamination detected, find outliers and plot the proportion
        of outliers per well.
        """
        # run sanity check to ensure step two was ran prior to step three
        if not hasattr(self, "partial_contamination_texture_detected"):
            raise RuntimeError(
                "You must run the second test before performing the third test."
            )

        print("Running step 3...")

        # check if partial contamination was detected
        if self.partial_contamination_texture_detected:
            print(
                "Finding outlier cells with anomalous texture around the nucleus..."
            )
            outliers = self.plot_proportion_outliers()

            # Calculate the 75th percentile (top 25%) of the outlier proportions
            top_25_percent_threshold = outliers["Proportion"].quantile(0.75)

            # Filter wells with proportions in the top 25%
            top_outlier_wells = outliers[
                outliers["Proportion"] >= top_25_percent_threshold
            ]
            # Print the wells in the top 25% horizontally
            print(f"\nNumber of wells in the top 25%: {len(top_outlier_wells)}")
            print("Wells in the top 25% of highest outlier proportions:")
            print(", ".join(top_outlier_wells["Well"]))
        else:
            print("No partial contamination detected; no outliers to find.")

    def detect_contamination(self) -> None:
        """
        Run all steps of the contamination detection process with conditional logic.
        """
        # Run the first step in the process to check for skewness and variability
        self.step_one_test()

        # Exit early if no skewness or variability detected
        if not self.is_skewed and not self.is_variable:
            return

        # Run the second step if skewness or variability was detected
        # Determines if the whole plate or partial plate is impacted
        self.step_two_test()

        # Only continue to step 3 if partial contamination texture was detected
        if (
            hasattr(self, "partial_contamination_texture_detected")
            and self.partial_contamination_texture_detected
        ):
            self.step_three_test()
