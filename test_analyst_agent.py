
import pandas as pd

from agents.analyst_agent import DataAnalystAgent


def test_dataset_dimensions():
    df = pd.DataFrame({
        "Name": ["A", "B", "C"],
        "Age": [21, 22, 23]
    })

    report = DataAnalystAgent().analyze_dataset(df)

    assert report["total_rows"] == 3
    assert report["total_columns"] == 2


def test_column_names():
    df = pd.DataFrame({
        "Name": ["A"],
        "Salary": [30000]
    })

    report = DataAnalystAgent().analyze_dataset(df)

    assert report["column_names"] == ["Name", "Salary"]


def test_data_types():
    df = pd.DataFrame({
        "Age": [21, 22],
        "Name": ["A", "B"]
    })

    report = DataAnalystAgent().analyze_dataset(df)

    assert report["data_types"]["Age"] == "int64"
    assert report["data_types"]["Name"] == str(df["Name"].dtype)


def test_missing_values():
    df = pd.DataFrame({
        "Age": [21, None, 23]
    })

    report = DataAnalystAgent().analyze_dataset(df)

    assert report["missing_values"]["Age"] == 1


def test_duplicate_rows():
    df = pd.DataFrame({
        "Name": ["A", "B", "A"],
        "Age": [21, 22, 21]
    })

    report = DataAnalystAgent().analyze_dataset(df)

    assert report["duplicate_rows"] == 1


def test_statistical_summary():
    df = pd.DataFrame({
        "Age": [20, 30, 40]
    })

    report = DataAnalystAgent().analyze_dataset(df)

    assert "Age" in report["statistical_summary"]
    assert report["statistical_summary"]["Age"]["mean"] == 30
