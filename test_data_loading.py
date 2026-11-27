
import pandas as pd
import pytest

from utils.data_processing import load_dataset


def test_load_csv(tmp_path):
    file_path = tmp_path / "sample.csv"

    df = pd.DataFrame({
        "Name": ["Alice", "Bob"],
        "Salary": [50000, 60000]
    })

    df.to_csv(file_path, index=False)

    result = load_dataset(file_path)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert result["Name"].tolist() == ["Alice", "Bob"]


def test_load_excel(tmp_path):
    file_path = tmp_path / "sample.xlsx"

    df = pd.DataFrame({
        "Name": ["Alice", "Bob"],
        "Salary": [50000, 60000]
    })

    df.to_excel(file_path, index=False)

    result = load_dataset(file_path)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert result["Salary"].tolist() == [50000, 60000]


def test_reject_unsupported_file():
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_dataset("sample.txt")
