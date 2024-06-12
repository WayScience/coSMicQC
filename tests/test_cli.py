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

    assert returncode == 0
    assert (
        stdout.strip()
        == """0    False
1    False
2    False
3    False
4    False
5    False
6    False
7    False
8     True
9     True
Name: Z_Score_example_feature, dtype: bool""".strip()
    )


def test_cli_find_outliers(basic_outlier_csv: str):
    """
    Test the `find_outliers` function of the CLI.
    """

    stdout, stderr, returncode = run_cli_command(
        (
            f"""cosmicqc find_outliers --df {basic_outlier_csv}"""
            """ --metadata_columns [] --feature_thresholds {"example_feature":1.0}"""
        )
    )

    assert returncode == 0
    assert (
        stdout.strip()
        == """Number of outliers: 2
Outliers Range:
example_feature Min: 9
example_feature Max: 10
   example_feature
8                9
9               10""".strip()
    )


def test_cli_label_outliers(basic_outlier_csv: str):
    """
    Test the `label_outliers` function of the CLI.
    """

    stdout, stderr, returncode = run_cli_command(
        (
            f"""cosmicqc label_outliers --df {basic_outlier_csv}"""
            """ --feature_thresholds {"example_feature":1.0}"""
        )
    )

    assert returncode == 0
    assert (
        stdout.strip()
        == """example_feature  outlier_custom
0                1           False
1                2           False
2                3           False
3                4           False
4                5           False
5                6           False
6                7           False
7                8           False
8                9            True
9               10            True""".strip()
    )
