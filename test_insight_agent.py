
from utils.data_processing import load_dataset
from agents.analyst_agent import DataAnalystAgent
from agents.insight_agent import InsightGenerationAgent

# Load sample dataset
data = load_dataset("sample.csv")

# Generate statistical analysis
analyst = DataAnalystAgent()
analysis_report = analyst.analyze_dataset(data)

# Initialize AI Insight Agent
insight_agent = InsightGenerationAgent()

# Generate AI insights
print("\nGenerating AI insights... Please wait.\n")

insights = insight_agent.generate_insights(analysis_report)

print("===== ANALYSTIQ AI INSIGHTS =====")
print(insights)
