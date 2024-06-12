"""
Utility functions for coSMicQC
"""

import sys
from typing import Optional

from functools import wraps

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
    else:
        return data

def df_to_string(func):
    """
    See https://github.com/google/python-fire/issues/274
    for why we need this (for now)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return str(func(*args, **kwargs))
    return wrapper
