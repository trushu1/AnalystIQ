
import pandas as pd


class DataAnalystAgent:

    def analyze_dataset(self, dataframe):
        """
        Analyze a dataset and generate a basic data report.
        """

        report = {}

        # 1. Dataset dimensions
        report["total_rows"] = dataframe.shape[0]
        report["total_columns"] = dataframe.shape[1]

        # 2. Column names
        report["column_names"] = dataframe.columns.tolist()

        # 3. Data types
        report["data_types"] = dataframe.dtypes.astype(str).to_dict()

        # 4. Missing values
        report["missing_values"] = dataframe.isnull().sum().to_dict()

        # 5. Duplicate records
        report["duplicate_rows"] = int(dataframe.duplicated().sum())

        # 6. Statistical summary
        report["statistical_summary"] = dataframe.describe(
        include="all"
        ).astype(object).fillna("").to_dict()

        return report