
import pandas as pd
from pathlib import Path


def load_dataset(file):
    """
    Load a CSV or Excel dataset.

    Args:
        file: Uploaded file or file path.

    Returns:
        pandas DataFrame containing the dataset.
    """

    file_name = file.name if hasattr(file, "name") else str(file)
    file_extension = Path(file_name).suffix.lower()

    if file_extension == ".csv":
        dataframe = pd.read_csv(file)

    elif file_extension in [".xlsx", ".xls"]:
        dataframe = pd.read_excel(file)

    else:
        raise ValueError(
            "Unsupported file format. Please upload a CSV or Excel file."
        )

    return dataframe

