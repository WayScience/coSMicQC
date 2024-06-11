import subprocess
import pytest
import json
import os
import pathlib
from .utils import run_cli_command



def test_cli_util():
    """
    Test the `identify_outliers` function of the CLI.
    """

    command = f"""echo 'hello world'"""
    stdout, stderr, returncode = run_cli_command(command)

    assert returncode == 0

def test_cli_identify_outliers(basic_outlier_csv):
    """
    Test the `identify_outliers` function of the CLI.
    """

    command = f"""cosmicqc identify_outliers --df {basic_outlier_csv} --feature_thresholds {"col1":1.0}"""
    stdout, stderr, returncode = run_cli_command(command)

    print(stderr)
    print(stdout)

    assert returncode == 0
    assert "outlier_custom" in stdout or "outlier_custom" in stderr

def test_cli_find_outliers(basic_outlier_csv):
    """
    Test the `find_outliers` function of the CLI.
    """

    command = f"""cosmicqc find_outliers {basic_outlier_csv} --metadata_columns '[\"col1\"]' --feature_thresholds '{"col1": 1.0}'"""
    stdout, stderr, returncode = run_cli_command(command)

    assert returncode == 0
    assert "outlier_custom" in stdout or "outlier_custom" in stderr

def test_cli_label_outliers(basic_outlier_csv):
    """
    Test the `label_outliers` function of the CLI.
    """

    command = f"""cosmicqc label_outliers {basic_outlier_csv} --feature_thresholds '{"col1": 1.0}'"""
    stdout, stderr, returncode = run_cli_command(command)

    assert returncode == 0
    assert "outlier_custom" in stdout or "outlier_custom" in stderr
