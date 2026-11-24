
from utils.data_processing import load_dataset
from agents.analyst_agent import DataAnalystAgent

# Load our sample dataset
data = load_dataset("sample.csv")

# Create the Data Analyst Agent
agent = DataAnalystAgent()

# Analyze the dataset
report = agent.analyze_dataset(data)

# Display the report
print("\n===== ANALYSTIQ DATA ANALYSIS REPORT =====")

for section, result in report.items():
    print(f"\n{section.upper()}:")
    print(result)
