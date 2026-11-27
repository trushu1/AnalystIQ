
import pandas as pd
import pytest

from utils.row_serializer import RowSerializer


@pytest.fixture
def serializer():
    return RowSerializer()


def test_serialize_dataframe(serializer):
    df = pd.DataFrame({
        "Name": ["Alice", "Bob"],
        "Department": ["IT", "HR"],
        "Salary": [60000, 40000]
    })

    records = serializer.serialize(df)

    assert len(records) == 2
    assert "Name: Alice" in records[0]["text"]
    assert "Department: HR" in records[1]["text"]


def test_preserves_row_index(serializer):
    df = pd.DataFrame(
        {"Name": ["Alice", "Bob"]},
        index=[10, 20]
    )

    records = serializer.serialize(df)

    assert records[0]["row_id"] == 10
    assert records[1]["row_id"] == 20


def test_handles_missing_values(serializer):
    df = pd.DataFrame({
        "Name": ["Alice", None],
        "Department": ["IT", "HR"]
    })

    records = serializer.serialize(df)

    assert "Not available" in records[1]["text"]


def test_empty_dataframe(serializer):
    df = pd.DataFrame()

    records = serializer.serialize(df)

    assert records == []


def test_rejects_invalid_input(serializer):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        serializer.serialize(["Alice", "Bob"])
