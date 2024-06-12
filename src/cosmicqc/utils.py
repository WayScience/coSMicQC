"""
Utility functions for coSMicQC
"""

import sys
from typing import Optional

from .scdataframe import SCDataFrame


def print_if_cli(data: SCDataFrame) -> Optional[SCDataFrame]:
    """
    Prints the data if the script is invoked from the cosmicqc command-line interface.

    Args:
        data (SCDataFrame):
            The data to print or return.

    Returns:
        Optional[SCDataFrame]:
            Returns the SCDataFrame if not run through CLI
            otherwise prints the SCDataFrame and returns None.
    """
    if any("bin/cosmicqc" in path for path in sys.argv):
        print(data)
    return data
