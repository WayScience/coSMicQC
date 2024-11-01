"""
Utilities for running pytest tests in coSMicQC
"""

import subprocess
from typing import Tuple


def run_cli_command(command: str) -> Tuple[str, str, int]:
    """
    Run a CLI command using subprocess and capture the output and return code.

    Args:
        command (list): The command to run as a list of strings.

    Returns:
        tuple: (str: stdout, str: stderr, int: returncode)
    """

    result = subprocess.run(
        command.split(" "), capture_output=True, text=True, check=False
    )
    return result.stdout, result.stderr, result.returncode
