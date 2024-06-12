"""
Setup coSMicQC CLI through python-fire
"""

import fire

from functools import wraps
from . import analyze

def cli_df_to_string(func: object) -> object:
    """
    See https://github.com/google/python-fire/issues/274
    for why we need this (for now)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import sys
        print(sys.argv)
        if any("bin/cosmicqc" in path for path in sys.argv):
            return str(func(*args, **kwargs))
        else:
            return func(*args, **kwargs)
    return wrapper

def cli_analyze() -> None:
    """
    Run the analyze module functions through python-fire CLI

    This function serves as the CLI entry point for functions
    within the analyze module.
    """

    for key, value in analyze.__dict__.items( ):
        if not key.startswith('_') and hasattr(value, '__call__' ):
            setattr(analyze, key, cli_df_to_string(value))

    fire.Fire(analyze)


if __name__ == "__main__":
    """
    Setup the CLI with python-fire for the coSMicQC package.

    This enables running the functions identify_outliers, find_outliers,
    and label_outliers from the command line interface through analyze
    """

    fire.Fire(analyze)
