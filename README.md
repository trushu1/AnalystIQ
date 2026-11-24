# AnalystIQ — AI-Powered Multi-Agent Data Analyst

An intelligent, AI-powered data analysis system that transforms raw datasets into meaningful insights using a coordinated team of specialized AI agents.

AnalystIQ helps users explore datasets, understand data patterns, generate visualizations, and ask questions about their data using natural language — without requiring extensive programming or SQL knowledge.

---

## Overview

Analyzing large datasets often requires technical knowledge, manual exploration, and time-consuming operations.

**AnalystIQ** simplifies this process through a multi-agent architecture where specialized agents collaborate to perform different data analysis tasks.

The system combines Python-based data processing with Large Language Model (LLM) capabilities to deliver structured analysis, visual insights, and natural-language explanations.

## Key Features

- **Dataset Upload:** Upload CSV and Excel files for analysis.
- **Automated Exploratory Data Analysis:** Understand dataset dimensions, column information, data types, and data quality.
- **Data Processing:** Perform structured data inspection and analysis using Pandas and NumPy.
- **Data Visualization:** Generate visual representations such as bar charts, line charts, pie charts, and scatter plots.
- **AI-Generated Insights:** Generate meaningful observations and summaries from the analyzed dataset.
- **Natural Language Querying:** Ask questions about uploaded data using everyday language.
- **AI-Powered Explanations:** Receive understandable explanations of query results.
- **Multi-Agent Coordination:** Coordinate specialized agents to complete different analytical tasks.
- **Interactive Dashboard:** Access analysis features through a Streamlit-based interface.

## Multi-Agent Architecture

AnalystIQ follows a modular multi-agent architecture. Each agent has a specialized responsibility and works with the other components to support the overall analysis workflow.

### Specialized Agents

| Agent | Responsibility |
|---|---|
| Data Analyst Agent | Examines datasets and performs analytical operations. |
| Visualization Agent | Supports the generation of data visualizations. |
| Insight Generation Agent | Produces meaningful observations and analytical summaries. |
| Natural Language Query Agent | Interprets user questions and prepares structured query plans. |
| Coordinator Agent | Coordinates the agents and manages the analysis workflow. |

### Architecture Workflow

```text
             User
               |
               v
       Streamlit Dashboard
               |
               v
       Coordinator Agent
               |
      +--------+---------+
      |        |         |
      v        v         v
   Analyst  Query    Visualization
   Agent    Agent       Agent
      |        |         |
      +--------+---------+
               |
               v
       Insight Generation
             Agent
               |
               v
       Results & Insights
               |
               v
       Interactive Dashboard
```

*Note: The diagram represents the logical responsibilities of the system.*

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Data Visualization | Plotly, Matplotlib |
| Generative AI | Groq API |
| LLM Integration | OpenAI GPT-OSS-120B |
| Environment Management | Python Virtual Environment |
| Configuration | Python Dotenv |
| Excel Support | OpenPyXL |

## Project Structure

```text
AnalystIQ/
│
├── agents/
│   ├── __init__.py
│   ├── analyst_agent.py
│   ├── coordinator_agent.py
│   ├── insight_agent.py
│   ├── query_agent.py
│   └── visualization_agent.py
│
├── utils/
│   ├── __init__.py
│   └── data_processing.py
│
├── app.py
├── requirements.txt
├── sample.csv
├── .gitignore
│
├── bar_chart.html
├── line_chart.html
├── pie_chart.html
├── scatter_chart.html
│
└── test_*.py
```

## Installation and Setup

Follow these steps to run AnalystIQ locally.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/AnalystIQ.git
cd AnalystIQ
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Groq API Key

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
```

Get your API key from [Groq Console](https://console.groq.com/).

**Important:** Never upload your `.env` file or expose your API key publicly.

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## How It Works

1. Upload a CSV or Excel dataset through the dashboard.
2. The system loads and processes the uploaded data.
3. The Data Analyst Agent examines the dataset.
4. The Visualization Agent supports graphical exploration.
5. The Insight Generation Agent generates analytical observations.
6. The Natural Language Query Agent interprets questions and prepares structured query plans.
7. The Coordinator Agent manages the workflow.
8. Results, explanations, and visual insights are presented through the dashboard.

## Sample Use Cases

AnalystIQ can support questions such as:

- What is the average salary in this dataset?
- Which department has the highest average salary?
- Show the top three highest-paid employees.
- How many employees belong to the IT department?
- What patterns can be observed in this dataset?

*Actual query support depends on the available dataset and implemented operations.*

## Security and Design Considerations

- API credentials are stored in environment variables.
- `.env` and virtual environment files are excluded through `.gitignore`.
- Data processing and supported analytical operations are handled using Python.
- Natural-language requests are converted into structured plans before supported operations are executed.

## Future Enhancements

- Advanced data cleaning and preprocessing.
- More sophisticated agent collaboration.
- Additional visualization options.
- Predictive analytics and forecasting.
- Exportable analytical reports.
- Enhanced conversational data exploration.
- Deployment through a cloud platform.

## Project Status

**Currently under development.**

AnalystIQ is being developed as a modular, AI-powered data analysis platform with a focus on multi-agent collaboration, natural-language interaction, and automated analytical insights.

## Author

**Trushna Amrutkar**

Computer Science and Engineering

GitHub: [@trushu1](https://github.com/trushu1)

---

<!-- *AnalystIQ — Making data analysis more accessible through AI.* -->