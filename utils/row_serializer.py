
import pandas as pd


class RowSerializer:
    """
    Converts DataFrame rows into structured text records
    for semantic search and RAG.
    """

    def serialize(self, dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if dataframe.empty:
            return []

        records = []

        for row_index, row in dataframe.iterrows():
            fields = []

            for column, value in row.items():
                if pd.isna(value):
                    value = "Not available"

                fields.append(f"{column}: {value}")

            records.append({
                "row_id": int(row_index) if isinstance(row_index, int) else str(row_index),
                "text": "\n".join(fields),
                "metadata": {
                    "source_index": str(row_index)
                }
            })

        return records
