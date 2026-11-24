
from utils.data_processing import load_dataset
from agents.visualization_agent import VisualizationAgent

# Load sample dataset
data = load_dataset("sample.csv")

# Create Visualization Agent
agent = VisualizationAgent()

# Generate charts
bar_chart = agent.generate_bar_chart(data, "Name", "Salary")
line_chart = agent.generate_line_chart(data, "Age", "Salary")
pie_chart = agent.generate_pie_chart(data, "Department")
scatter_chart = agent.generate_scatter_chart(data, "Age", "Salary")

# Verify chart generation
print("Bar chart:", bar_chart is not None)
print("Line chart:", line_chart is not None)
print("Pie chart:", pie_chart is not None)
print("Scatter chart:", scatter_chart is not None)

# Save charts as HTML files
bar_chart.write_html("bar_chart.html")
line_chart.write_html("line_chart.html")
pie_chart.write_html("pie_chart.html")
scatter_chart.write_html("scatter_chart.html")

print("\nAll four charts generated successfully!")
