
import pandas as pd
import pytest

from agents.coordinator_agent import CoordinatorAgent


def test_coordinator_returns_expected_results(monkeypatch):
    df = pd.DataFrame({
        "Department": ["IT", "HR", "IT"],
        "Salary": [50000, 40000, 60000],
        "Experience": [2, 3, 5]
    })

    coordinator = CoordinatorAgent()

    # Avoid making a real Groq API call
    monkeypatch.setattr(
        coordinator.insight_agent,
        "generate_insights",
        lambda report: "Test insights generated"
    )

    results = coordinator.analyze(df)

    assert "analysis_report" in results
    assert "charts" in results
    assert "insights" in results

    assert results["analysis_report"]["total_rows"] == 3
    assert results["analysis_report"]["total_columns"] == 3
    assert results["insights"] == "Test insights generated"


def test_coordinator_generates_expected_charts(monkeypatch):
    df = pd.DataFrame({
        "Department": ["IT", "HR", "IT"],
        "Salary": [50000, 40000, 60000],
        "Experience": [2, 3, 5]
    })

    coordinator = CoordinatorAgent()

    monkeypatch.setattr(
        coordinator.insight_agent,
        "generate_insights",
        lambda report: "Test insights"
    )

    results = coordinator.analyze(df)

    assert "bar_chart" in results["charts"]
    assert "pie_chart" in results["charts"]
    assert "scatter_chart" in results["charts"]
    assert "line_chart" in results["charts"]


def test_coordinator_rejects_empty_dataset():
    coordinator = CoordinatorAgent()

    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="uploaded dataset is empty"):
        coordinator.analyze(empty_df)
