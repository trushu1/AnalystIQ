
from unittest.mock import MagicMock

from agents.insight_agent import InsightGenerationAgent


def test_generate_insights_returns_ai_response():
    agent = InsightGenerationAgent()

    # Mock the Groq API response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        "Key Findings: Sales are increasing."
    )

    agent.client.chat.completions.create = MagicMock(
        return_value=mock_response
    )

    analysis_report = {
        "total_rows": 3,
        "total_columns": 2,
        "column_names": ["Product", "Sales"]
    }

    result = agent.generate_insights(analysis_report)

    assert isinstance(result, str)
    assert "Key Findings" in result
    agent.client.chat.completions.create.assert_called_once()


def test_generate_insights_handles_empty_report():
    agent = InsightGenerationAgent()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        "No meaningful insights available."
    )

    agent.client.chat.completions.create = MagicMock(
        return_value=mock_response
    )

    result = agent.generate_insights({})

    assert isinstance(result, str)
    assert len(result) > 0
