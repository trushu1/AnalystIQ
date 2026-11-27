
from agents.analyst_agent import DataAnalystAgent
from agents.visualization_agent import VisualizationAgent
from agents.insight_agent import InsightGenerationAgent


class CoordinatorAgent:

    def __init__(self):
        self.analyst = DataAnalystAgent()
        self.visualizer = VisualizationAgent()
        self.insight_agent = InsightGenerationAgent()

        # RAG components are initialized only when requested.
        self.retrieval_agent = None
        self.rag_pipeline = None

    def analyze(self, dataframe):
        """
        Coordinate existing dataset analysis and visualization agents.
        """

        if dataframe is None or dataframe.empty:
            raise ValueError("The uploaded dataset is empty.")

        print("Coordinator: Starting data analysis...")
        analysis_report = self.analyst.analyze_dataset(dataframe)

        print("Coordinator: Generating visualizations...")

        charts = {}

        numeric_columns = dataframe.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = dataframe.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns.tolist()

        if categorical_columns and numeric_columns:
            charts["bar_chart"] = self.visualizer.generate_bar_chart(
                dataframe,
                categorical_columns[0],
                numeric_columns[0]
            )

        if categorical_columns:
            charts["pie_chart"] = self.visualizer.generate_pie_chart(
                dataframe,
                categorical_columns[0]
            )

        if len(numeric_columns) >= 2:
            charts["scatter_chart"] = self.visualizer.generate_scatter_chart(
                dataframe,
                numeric_columns[0],
                numeric_columns[1]
            )

            charts["line_chart"] = self.visualizer.generate_line_chart(
                dataframe,
                numeric_columns[0],
                numeric_columns[1]
            )

        print("Coordinator: Generating AI insights...")

        insights = self.insight_agent.generate_insights(analysis_report)

        print("Coordinator: Analysis completed!")

        return {
            "analysis_report": analysis_report,
            "charts": charts,
            "insights": insights
        }

    def build_retrieval_index(
        self,
        dataframe,
        embedding_service=None,
        groq_client=None
    ):
        """
        Build a semantic search index for the uploaded dataset.
        """

        if dataframe is None or dataframe.empty:
            raise ValueError("Cannot index an empty dataset.")

        from agents.retrieval_agent import RetrievalAgent
        from utils.rag_pipeline import RAGPipeline

        retrieval_agent = RetrievalAgent(
            embedding_service=embedding_service
        )

        index_result = retrieval_agent.build_index(dataframe)

        rag_pipeline = RAGPipeline(
            retrieval_agent=retrieval_agent,
            client=groq_client
        )

        self.retrieval_agent = retrieval_agent
        self.rag_pipeline = rag_pipeline

        return index_result

    def answer_semantic_question(self, question, top_k=5):
        """
        Answer a natural-language question using RAG.
        """

        if self.rag_pipeline is None:
            raise RuntimeError(
                "Retrieval index not initialized. Build the index first."
            )

        return self.rag_pipeline.answer_question(
            question,
            top_k=top_k
        )
