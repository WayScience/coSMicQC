"""
Setup coSMicQC CLI through python-fire
"""

import fire
from . import analyze

def cli_analyze():
    """
    Run the analyze module functions through python-fire CLI

    This function serves as the CLI entry point for functions
    within the analyze module.
    """
    fire.Fire(analyze)

if __name__ == "__main__":
    """
    Setup the CLI with python-fire for the coSMicQC package.

    This enables running the functions identify_outliers, find_outliers,
    and label_outliers from the command line interface through analyze
    """

    fire.Fire(analyze)
