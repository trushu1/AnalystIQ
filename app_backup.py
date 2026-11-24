
import streamlit as st
import pandas as pd

from agents.coordinator_agent import CoordinatorAgent


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AnalystIQ - AI Data Analyst",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.title("📊 AnalystIQ")
st.subheader("AI-Powered Multi-Agent Data Analysis System")

st.markdown(
    "Upload your dataset and let our AI agents analyze "
    "your data, generate visualizations, and discover insights."
)

st.divider()


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"]
)


# --------------------------------------------------
# INITIALIZE COORDINATOR
# --------------------------------------------------

@st.cache_resource
def get_coordinator():
    return CoordinatorAgent()


# --------------------------------------------------
# PROCESS DATASET
# --------------------------------------------------

if uploaded_file is not None:

    try:
        # Load dataset
        if uploaded_file.name.lower().endswith(".csv"):
            dataframe = pd.read_csv(uploaded_file)
        else:
            dataframe = pd.read_excel(uploaded_file)

        if dataframe.empty:
            st.error("The uploaded dataset is empty.")

        else:

            st.success("Dataset uploaded successfully!")

            # Dataset preview
            st.header("📁 Dataset Preview")
            st.dataframe(dataframe.head(10), use_container_width=True)

            # Dataset metrics
            st.header("📈 Dataset Overview")

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

            st.divider()

            # Run analysis
            if st.button("🚀 Analyze Dataset", type="primary"):

                with st.spinner(
                    "Our AI agents are analyzing your dataset..."
                ):

                    coordinator = get_coordinator()
                    results = coordinator.analyze(dataframe)

                st.session_state["analysis_results"] = results

                st.success("Analysis completed successfully!")

            # Display analysis results
            if "analysis_results" in st.session_state:

                results = st.session_state["analysis_results"]

                # Statistical report
                st.header("🔍 Data Analysis Report")

                report = results["analysis_report"]

                st.write("**Column Names:**", report["column_names"])

                st.write("**Data Types:**")
                st.json(report["data_types"])

                st.write("**Missing Values:**")
                st.json(report["missing_values"])

                st.write("**Statistical Summary:**")
                st.dataframe(
                    pd.DataFrame(report["statistical_summary"]),
                    use_container_width=True
                )

                # Visualizations
                st.header("📊 Data Visualizations")

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

                else:
                    st.info(
                        "Not enough suitable columns to generate charts."
                    )

                # AI insights
                st.header("🤖 AI-Generated Insights")

                st.markdown(results["insights"])

    except Exception as error:

        st.error(f"Something went wrong: {error}")


else:

    st.info(
        "Please upload a CSV or Excel file to get started."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AnalystIQ | Multi-Agent AI Data Analysis System"
)
