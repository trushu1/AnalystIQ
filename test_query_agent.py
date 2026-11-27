
import pandas as pd
import pytest
from unittest.mock import MagicMock

from agents.query_agent import NaturalLanguageQueryAgent


@pytest.fixture
def agent():
    return NaturalLanguageQueryAgent()


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "Name": ["A", "B", "C", "D"],
        "Department": ["IT", "HR", "IT", "Sales"],
        "Salary": [50000, 40000, 60000, 45000],
        "Age": [25, 30, 28, 22]
    })


def test_mean(agent, sample_data):
    result = agent.execute_plan(
        {"operation": "mean", "column": "Salary"},
        sample_data
    )
    assert result == 48750.0


def test_sum(agent, sample_data):
    result = agent.execute_plan(
        {"operation": "sum", "column": "Salary"},
        sample_data
    )
    assert result == 195000.0


def test_count_rows(agent, sample_data):
    result = agent.execute_plan(
        {"operation": "count_rows"},
        sample_data
    )
    assert result == 4


def test_filtered_count(agent, sample_data):
    result = agent.execute_plan(
        {
            "operation": "count_rows",
            "filter_column": "Department",
            "filter_value": "IT"
        },
        sample_data
    )
    assert result == 2


def test_value_counts(agent, sample_data):
    result = agent.execute_plan(
        {"operation": "value_counts", "column": "Department"},
        sample_data
    )
    assert result["IT"] == 2


def test_groupby_aggregate(agent, sample_data):
    result = agent.execute_plan(
        {
            "operation": "groupby_aggregate",
            "group_column": "Department",
            "value_column": "Salary",
            "aggregation": "mean"
        },
        sample_data
    )
    assert result["IT"] == 55000.0


def test_top_n(agent, sample_data):
    result = agent.execute_plan(
        {
            "operation": "top_n",
            "column": "Salary",
            "n": 2,
            "ascending": False
        },
        sample_data
    )
    assert len(result) == 2
    assert result.iloc[0]["Salary"] == 60000


def test_invalid_operation(agent, sample_data):
    with pytest.raises(ValueError, match="Unsupported operation"):
        agent.execute_plan(
            {"operation": "delete"},
            sample_data
        )


def test_invalid_column(agent, sample_data):
    with pytest.raises(ValueError, match="Invalid column"):
        agent.execute_plan(
            {"operation": "mean", "column": "Bonus"},
            sample_data
        )


def test_non_numeric_column(agent, sample_data):
    with pytest.raises(ValueError, match="numeric column"):
        agent.execute_plan(
            {"operation": "mean", "column": "Name"},
            sample_data
        )


def test_interpret_question(agent, sample_data, monkeypatch):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        '{"operation": "mean", "column": "Salary"}'
    )

    monkeypatch.setattr(
        agent.client.chat.completions,
        "create",
        MagicMock(return_value=mock_response)
    )

    result = agent.interpret_question(
        "What is the average salary?",
        sample_data
    )

    assert result["operation"] == "mean"
    assert result["column"] == "Salary"


def test_explain_result(agent, monkeypatch):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        "The average salary is 48750."
    )

    monkeypatch.setattr(
        agent.client.chat.completions,
        "create",
        MagicMock(return_value=mock_response)
    )

    result = agent.explain_result(
        "What is the average salary?",
        48750.0
    )

    assert isinstance(result, str)
    assert "48750" in result


def test_answer_question(agent, sample_data, monkeypatch):
    monkeypatch.setattr(
        agent,
        "interpret_question",
        lambda question, df: {
            "operation": "mean",
            "column": "Salary"
        }
    )

    monkeypatch.setattr(
        agent,
        "explain_result",
        lambda question, result: "Average salary is 48750."
    )

    result = agent.answer_question(
        "What is the average salary?",
        sample_data
    )

    assert result["result"] == 48750.0
    assert result["explanation"] == "Average salary is 48750."
    assert "plan" in result
