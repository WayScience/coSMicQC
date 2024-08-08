# SBI2 2024 Abstract Submission - coSMicQC

## Author(s):

Dave Bunten, Jenna Tomkinson, Gregory Way

## Title (under 100 characters):

Single-cell Morphology Quality Control (coSMicQC)

## Abstract (under 500 words):

High-dimensional single-cell morphology data from large-scale microscopy drug screening applications help prioritize effective treatments for patients suffering from various diseases and enable the discovery of new biological mechanisms. Image analysis pipelines to process these single-cell data often introduce errors in the segmentation step, where software improperly segments single cells (for example, capturing undersized or overly large portions of a cell compartment) and incorrectly identifies artifacts like dust or other debris as single cells. These errors lead to erroneous single-cell measurements which need to be removed prior to single-cell analyses to help ensure accurate results. Research scientists often use bespoke approaches to filter single cells or aggregate all single cells into bulk profiles, reducing errors' impact and preventing single-cell analysis. This leads to duplicated effort, human error, and a lack of quality control in single-cell feature data which overall may result in a reduced potential for discoveries or inaccurate outcomes.

We introduce a Python package called coSMicQC (Single cell Morphology Quality Control) to improve single-cell morphology analysis. coSMicQC uses high-content morphology measurements to define default thresholds for removing single cells and enables users to customize quality control parameters. Accessible through both command line interface (CLI) and Python application programming interface (API), coSMicQC seamlessly integrates into diverse analytical workflows, including standalone scripts/workflows to interactive Jupyter Notebooks. Notably, the package creates interactive and exportable visualizations that illustrate outlier distributions. The backbone of this package is a novel data format: the CytoDataFrame. CytoDataFrames facilitate real-time exploration of single-cell images within any pandas environment to seamlessly link single-cell morphology measurements with single-cell images. Leveraging real-world datasets, including Joint Undertaking in Morphological Profiling (CPJUMP1) data, we show how our tool empowers researchers to identify technical outliers within single-cell profile features and improve single-cell analysis.
