
import pandas as pd
import plotly.graph_objects as go

from agents.visualization_agent import VisualizationAgent


def test_generate_bar_chart():
    df = pd.DataFrame({
        "Department": ["IT", "HR", "Sales"],
        "Salary": [50000, 40000, 60000]
    })

    agent = VisualizationAgent()
    chart = agent.generate_bar_chart(df, "Department", "Salary")

    assert isinstance(chart, go.Figure)
    assert len(chart.data) > 0


def test_generate_line_chart():
    df = pd.DataFrame({
        "Age": [22, 25, 30],
        "Salary": [30000, 45000, 60000]
    })

    agent = VisualizationAgent()
    chart = agent.generate_line_chart(df, "Age", "Salary")

    assert isinstance(chart, go.Figure)
    assert len(chart.data) > 0


def test_generate_pie_chart():
    df = pd.DataFrame({
        "Department": ["IT", "HR", "IT", "Sales"]
    })

    agent = VisualizationAgent()
    chart = agent.generate_pie_chart(df, "Department")

    assert isinstance(chart, go.Figure)
    assert len(chart.data) > 0


def test_generate_scatter_chart():
    df = pd.DataFrame({
        "Age": [22, 25, 30],
        "Salary": [30000, 45000, 60000]
    })

    agent = VisualizationAgent()
    chart = agent.generate_scatter_chart(df, "Age", "Salary")

    assert isinstance(chart, go.Figure)
    assert len(chart.data) > 0
