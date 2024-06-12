from .utils import run_cli_command


def test_cli_util():
    """
    Test the `identify_outliers` function of the CLI.
    """

    command = """echo 'hello world'"""
    stdout, stderr, returncode = run_cli_command(command)

    assert returncode == 0


def test_cli_identify_outliers(basic_outlier_csv: str):
    """
    Test the `identify_outliers` function of the CLI.
    """

    stdout, stderr, returncode = run_cli_command(
        (
            f"""cosmicqc identify_outliers --df {basic_outlier_csv}"""
            """ --feature_thresholds {"example_feature":1.0}"""
        )
    )

    print(stderr)
    print(stdout)

    assert returncode == 0
    assert "outlier_custom" in stdout or "outlier_custom" in stderr


def test_cli_find_outliers(basic_outlier_csv: str):
    """
    Test the `find_outliers` function of the CLI.
    """

    stdout, stderr, returncode = run_cli_command(
        (
            f"""cosmicqc find_outliers {basic_outlier_csv}"""
            """ --metadata_columns '[\"col1\"]' --feature_thresholds '{"example_feature": 1.0}'"""
        )
    )

    assert returncode == 0
    assert "outlier_custom" in stdout or "outlier_custom" in stderr


def test_cli_label_outliers(basic_outlier_csv: str):
    """
    Test the `label_outliers` function of the CLI.
    """

    stdout, stderr, returncode = run_cli_command(
        (
            f"""cosmicqc label_outliers {basic_outlier_csv}"""
            """ --feature_thresholds '{"example_feature": 1.0}'"""
        )
    )

    assert returncode == 0
    assert "outlier_custom" in stdout or "outlier_custom" in stderr
