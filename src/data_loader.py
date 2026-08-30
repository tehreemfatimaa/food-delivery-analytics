import os
import glob
import pandas as pd


def find_dataset_path(data_dir="data"):
    """
    Search for CSV dataset files in the project folder.
    First checks the 'data/' folder, then checks the root folder.

    Returns:
        str: Path to the discovered CSV file.

    Raises:
        FileNotFoundError: If no CSV file is found.
    """
    # Look for CSV files in the data directory
    data_csvs = glob.glob(os.path.join(data_dir, "*.csv"))
    if data_csvs:
        return data_csvs[0]

    # Look for CSV files in the root directory
    root_csvs = glob.glob("*.csv")
    if root_csvs:
        return root_csvs[0]

    raise FileNotFoundError(
        "No CSV dataset found in 'data/' directory or project root folder. "
        "Please place your food delivery CSV dataset inside the 'data/' directory."
    )


def load_raw_data(file_path=None):
    """
    Load the food delivery dataset into a Pandas DataFrame.

    Args:
        file_path (str, optional): Path to the CSV file. If None, auto-discovers dataset.

    Returns:
        tuple: (pd.DataFrame, str) DataFrame and file path used.
    """
    if file_path is None:
        file_path = find_dataset_path()

    print(f"Loading dataset from: {file_path}")

    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)

    return df, file_path


def inspect_raw_data(df):
    """
    Inspect raw dataset statistics including shape, data types, missing values,
    and duplicate records.

    Args:
        df (pd.DataFrame): The raw DataFrame.

    Returns:
        dict: A dictionary containing inspection metrics.
    """
    # Calculate basic dataset metadata
    total_rows, total_cols = df.shape
    missing_values = df.isnull().sum().to_dict()
    duplicate_count = int(df.duplicated().sum())

    # Build summary dictionary
    inspection_summary = {
        "num_rows": total_rows,
        "num_cols": total_cols,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": missing_values,
        "duplicate_count": duplicate_count,
        "first_5_rows": df.head().to_dict(orient="records"),
    }

    return inspection_summary
