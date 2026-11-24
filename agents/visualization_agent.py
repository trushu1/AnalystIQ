
import pandas as pd
import plotly.express as px


class VisualizationAgent:

    def generate_bar_chart(self, dataframe, x_column, y_column):
        """Generate a bar chart."""
        fig = px.bar(
            dataframe,
            x=x_column,
            y=y_column,
            title=f"{y_column} by {x_column}"
        )
        return fig

    def generate_line_chart(self, dataframe, x_column, y_column):
        """Generate a line chart."""
        fig = px.line(
            dataframe,
            x=x_column,
            y=y_column,
            title=f"{y_column} Trend"
        )
        return fig

    def generate_pie_chart(self, dataframe, category_column):
        """Generate a pie chart."""
        counts = dataframe[category_column].value_counts().reset_index()
        counts.columns = [category_column, "Count"]

        fig = px.pie(
            counts,
            names=category_column,
            values="Count",
            title=f"Distribution of {category_column}"
        )
        return fig

    def generate_scatter_chart(self, dataframe, x_column, y_column):
        """Generate a scatter plot."""
        fig = px.scatter(
            dataframe,
            x=x_column,
            y=y_column,
            title=f"{y_column} vs {x_column}"
        )
        return fig
