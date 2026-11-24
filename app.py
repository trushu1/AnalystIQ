
import streamlit as st
import pandas as pd
import hashlib
from io import BytesIO

from agents.coordinator_agent import CoordinatorAgent
from agents.query_agent import NaturalLanguageQueryAgent

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AnalystIQ | AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] * {
    color: white;
}

h1, h2, h3 {
    color: #1e293b;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# COORDINATOR INITIALIZATION
# --------------------------------------------------

@st.cache_resource
def get_coordinator():
    return CoordinatorAgent()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("📊 AnalystIQ")

    st.caption("AI-Powered Data Intelligence")

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "📁 Data Explorer",
        "📊 Visualizations",
        "🤖 AI Insights",
        "💬 Ask AnalystIQ"
    ],
    label_visibility="collapsed"
)

    st.divider()

    st.subheader("Upload Dataset")

    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )

    st.divider()

    st.caption("Powered by Python, Streamlit and Groq AI")


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

if uploaded_file is None:

    st.title("Welcome to AnalystIQ 👋")

    st.markdown(
        "Your intelligent multi-agent data analysis assistant."
    )

    st.info(
        "Upload a CSV or Excel dataset from the sidebar "
        "to begin your analysis."
    )

    st.markdown("### What AnalystIQ can do")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔍")
        st.write("**Data Analysis**")
        st.caption("Explore data, statistics and quality.")

    with col2:
        st.markdown("### 📈")
        st.write("**Visualizations**")
        st.caption("Generate interactive data charts.")

    with col3:
        st.markdown("### 🧠")
        st.write("**AI Insights**")
        st.caption("Discover patterns and recommendations.")

    st.stop()


# --------------------------------------------------
# READ UPLOADED FILE
# --------------------------------------------------

try:

    file_bytes = uploaded_file.getvalue()

    file_signature = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.get("file_signature") != file_signature:

        st.session_state["file_signature"] = file_signature

        st.session_state.pop("analysis_results", None)

    if uploaded_file.name.lower().endswith(".csv"):

        dataframe = pd.read_csv(BytesIO(file_bytes))

    else:

        dataframe = pd.read_excel(BytesIO(file_bytes))

    if dataframe.empty:

        st.error("The uploaded dataset is empty.")
        st.stop()

except Exception as error:

    st.error(f"Unable to read the uploaded file: {error}")
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📊 AnalystIQ Dashboard")

st.markdown(
    "Analyze your data, discover patterns and generate "
    "AI-powered business insights."
)

st.caption(f"Current Dataset: {uploaded_file.name}")

st.divider()


# --------------------------------------------------
# DATASET METRICS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Rows", dataframe.shape[0])

with col2:
    st.metric("Total Columns", dataframe.shape[1])

with col3:
    st.metric(
        "Missing Values",
        int(dataframe.isnull().sum().sum())
    )

with col4:
    st.metric(
        "Duplicate Rows",
        int(dataframe.duplicated().sum())
    )


# --------------------------------------------------
# ANALYSIS BUTTON
# --------------------------------------------------

if "analysis_results" not in st.session_state:

    st.info(
        "Your dataset is ready. Run the AI agents to generate "
        "charts, analysis and insights."
    )

    if st.button("🚀 Run AI Analysis", type="primary"):

        try:

            with st.spinner(
                "Our AI agents are working on your dataset..."
            ):

                coordinator = get_coordinator()

                results = coordinator.analyze(dataframe)

                st.session_state["analysis_results"] = results

            st.success("Analysis completed successfully!")

            st.rerun()

        except Exception as error:

            st.error(f"Analysis failed: {error}")


# --------------------------------------------------
# GET ANALYSIS RESULTS
# --------------------------------------------------

results = st.session_state.get("analysis_results")


# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "🏠 Dashboard":

    st.header("🏠 Dashboard Overview")

    st.write(
        "Welcome to your data intelligence workspace."
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        dataframe.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.write("**Column Names**")

        st.write(dataframe.columns.tolist())

    with info_col2:

        st.write("**Data Types**")

        st.write(
            dataframe.dtypes.astype(str).to_dict()
        )

    if results:

        st.success("AI analysis is ready!")

        st.write(
            "Navigate to Visualizations or AI Insights "
            "using the sidebar."


        )


# --------------------------------------------------
# DATA EXPLORER PAGE
# --------------------------------------------------

elif page == "📁 Data Explorer":

    st.header("📁 Data Explorer")

    st.subheader("Complete Dataset")

    st.dataframe(
        dataframe,
        use_container_width=True
    )

    st.subheader("Data Types")

    st.write(
        dataframe.dtypes.astype(str).to_dict()
    )

    st.subheader("Missing Values")

    missing_data = dataframe.isnull().sum()

    st.dataframe(
        missing_data.rename("Missing Count"),
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        dataframe.describe(include="all"),
        use_container_width=True
    )

    if results:

        st.subheader("Duplicate Records")

        st.write(
            results["analysis_report"]["duplicate_rows"]
        )


# --------------------------------------------------
# VISUALIZATION PAGE
# --------------------------------------------------

elif page == "📊 Visualizations":

    st.header("📊 Interactive Visualizations")

    if results:

        charts = results["charts"]

        if charts:

            for chart_name, chart in charts.items():

                st.subheader(
                    chart_name.replace("_", " ").title()
                )

                st.plotly_chart(
                    chart,
                    use_container_width=True
                )

                st.divider()

        else:

            st.warning(
                "No suitable charts could be generated "
                "for this dataset."
            )

    else:

        st.info(
            "Run AI Analysis from the Dashboard first "
            "to generate visualizations."
        )


# --------------------------------------------------
# AI INSIGHTS PAGE
# --------------------------------------------------

elif page == "🤖 AI Insights":

    st.header("🤖 AI-Generated Insights")

    if results:

        st.markdown(results["insights"])

    else:

        st.info(
            "Run AI Analysis from the Dashboard first "
            "to generate insights."
        )

# --------------------------------------------------
# NATURAL LANGUAGE QUERY PAGE
# --------------------------------------------------

elif page == "💬 Ask AnalystIQ":

    st.header("💬 Ask AnalystIQ")

    st.markdown(
        "Ask questions about your uploaded dataset "
        "in simple English."
    )

    st.info(
        "Example: What is the average salary?"
    )

    # Initialize Query Agent
    @st.cache_resource
    def get_query_agent():
        return NaturalLanguageQueryAgent()

    query_agent = get_query_agent()

    # User question
    question = st.text_input(
        "Enter your question",
        placeholder="Example: Which department has the highest average salary?"
    )

    if st.button("🔍 Get Answer", type="primary"):

        if question.strip():

            try:

                with st.spinner("AnalystIQ is thinking..."):

                    answer = query_agent.answer_question(
                        question,
                        dataframe
                    )

                st.success("Answer generated successfully!")

                st.subheader("Your Question")
                st.write(answer["question"])

                st.subheader("AI Interpreted Plan")
                st.json(answer["plan"])

                st.subheader("Result")

                if isinstance(answer["result"], pd.DataFrame):
                    st.dataframe(
                        answer["result"],
                        use_container_width=True
                    )
                else:
                    st.write(answer["result"])


                st.subheader("AI Explanation")
                st.write(answer["explanation"])

            except Exception as error:

                st.error(f"Unable to answer your question: {error}")

        else:

            st.warning("Please enter a question first.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AnalystIQ | Multi-Agent AI Data Analysis System"
)

st.caption(
    "Built with Python • Streamlit • Pandas • Plotly • Groq AI"
)
