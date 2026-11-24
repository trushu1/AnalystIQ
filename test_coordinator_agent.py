from utils.data_processing import load_dataset
from agents.coordinator_agent import CoordinatorAgent

data = load_dataset("sample.csv")

coordinator = CoordinatorAgent()
results = coordinator.analyze(data)

print("\n===== DATA ANALYSIS REPORT =====")
print(results["analysis_report"])

print("\n===== GENERATED CHARTS =====")
for chart_name in results["charts"]:
    print(chart_name)

print("\n===== AI GENERATED INSIGHTS =====")
print(results["insights"])

print("\n===== ANALYSTIQ WORKFLOW SUCCESSFUL! =====")
