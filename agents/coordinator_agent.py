
from agents.analyst_agent import DataAnalystAgent
from agents.visualization_agent import VisualizationAgent
from agents.insight_agent import InsightGenerationAgent


class CoordinatorAgent:

    def __init__(self):
        self.analyst = DataAnalystAgent()
        self.visualizer = VisualizationAgent()
        self.insight_agent = InsightGenerationAgent()

    def analyze(self, dataframe):
        """
        Coordinate all agents to analyze a dataset,
        generate visualizations, and produce AI insights.
        """

        if dataframe is None or dataframe.empty:
            raise ValueError("The uploaded dataset is empty.")

        # Step 1: Analyze the dataset
        print("Coordinator: Starting data analysis...")
        analysis_report = self.analyst.analyze_dataset(dataframe)

        # Step 2: Generate visualizations
        print("Coordinator: Generating visualizations...")

        charts = {}

        numeric_columns = dataframe.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = dataframe.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns.tolist()

        # Bar chart: first categorical column vs first numeric column
        if categorical_columns and numeric_columns:
            charts["bar_chart"] = self.visualizer.generate_bar_chart(
                dataframe,
                categorical_columns[0],
                numeric_columns[0]
            )

        # Pie chart: first categorical column
        if categorical_columns:
            charts["pie_chart"] = self.visualizer.generate_pie_chart(
                dataframe,
                categorical_columns[0]
            )

        # Scatter chart: first two numeric columns
        if len(numeric_columns) >= 2:
            charts["scatter_chart"] = self.visualizer.generate_scatter_chart(
                dataframe,
                numeric_columns[0],
                numeric_columns[1]
            )

        # Line chart: first two numeric columns
        if len(numeric_columns) >= 2:
            charts["line_chart"] = self.visualizer.generate_line_chart(
                dataframe,
                numeric_columns[0],
                numeric_columns[1]
            )

        # Step 3: Generate AI insights
        print("Coordinator: Generating AI insights...")

        insights = self.insight_agent.generate_insights(analysis_report)

        # Step 4: Collect all results
        print("Coordinator: Analysis completed!")

        return {
            "analysis_report": analysis_report,
            "charts": charts,
            "insights": insights
        }
