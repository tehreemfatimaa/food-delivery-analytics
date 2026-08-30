import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth in kilometers
    using the Haversine formula.

    Args:
        lat1, lon1: Coordinates of point 1 (degrees)
        lat2, lon2: Coordinates of point 2 (degrees)

    Returns:
        np.ndarray or float: Distance in kilometers
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [np.abs(lat1), np.abs(lon1), np.abs(lat2), np.abs(lon2)])

    # Haversine formula calculations
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0  # Earth's radius in kilometers
    return r * c


def clean_dataset(df):
    """
    Clean and prepare the food delivery dataset for analysis.

    Cleaning Steps:
    1. Standardize column names (map Kaggle names to clean canonical names).
    2. Handle string column white spaces and prefixes (e.g., 'conditions Sunny' -> 'Sunny').
    3. Clean numeric columns (convert Time_taken (min), Delivery_person_Age, Delivery_person_Ratings to numeric).
    4. Compute or clean Delivery Distance in kilometers.
    5. Handle exact duplicates and missing values safely.

    Args:
        df (pd.DataFrame): Raw food delivery DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying original data inplace
    cleaned_df = df.copy()

    # Step 1: Remove leading/trailing spaces from column headers
    cleaned_df.columns = cleaned_df.columns.str.strip()

    # Flexible dynamic column mapping dictionary
    # Maps any variation of Kaggle / custom column headers to actual names
    canonical_map = {}
    for col in cleaned_df.columns:
        col_clean = col.strip().lower().replace(" ", "_")
        
        if "time_taken" in col_clean or "delivery_time" in col_clean or "duration" in col_clean:
            canonical_map[col] = "Time_taken (min)"
        elif "traffic" in col_clean:
            canonical_map[col] = "Road_traffic_density"
        elif "weather" in col_clean:
            canonical_map[col] = "Weather_conditions"
        elif "age" in col_clean:
            canonical_map[col] = "Delivery_person_Age"
        elif "rating" in col_clean:
            canonical_map[col] = "Delivery_person_Ratings"
        elif "distance" in col_clean:
            canonical_map[col] = "distance_km"
        elif "city" in col_clean:
            canonical_map[col] = "City"
        elif "vehicle_condition" in col_clean:
            canonical_map[col] = "Vehicle_condition"

    # Apply canonical mapping
    cleaned_df.rename(columns=canonical_map, inplace=True)

    # Step 2: Clean string columns (strip whitespace, clean prefixes)
    string_cols = cleaned_df.select_dtypes(include=["object"]).columns
    for col in string_cols:
        # Convert to string and strip trailing/leading spaces
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

        # Remove "conditions " prefix in Weather_conditions column if present
        if col == "Weather_conditions":
            cleaned_df[col] = cleaned_df[col].str.replace("conditions ", "", regex=False)

        # Replace literal 'nan', 'NaN', 'null', 'NaN ', 'NaN' strings with actual np.nan
        cleaned_df[col] = cleaned_df[col].replace(
            ["nan", "NaN", "null", "NaN ", "NaN", "NAN", "None", ""], np.nan
        )

    # Step 3: Clean Time_taken (min) column (extract numeric digits)
    if "Time_taken (min)" in cleaned_df.columns:
        # Handle values like "(min) 24", "24 min", "24.0"
        extracted = (
            cleaned_df["Time_taken (min)"]
            .astype(str)
            .str.extract(r"(\d+\.?\d*)")[0]
        )
        cleaned_df["Time_taken (min)"] = pd.to_numeric(extracted, errors="coerce")

    # Step 4: Clean Delivery_person_Age, Delivery_person_Ratings, and Vehicle_condition numeric columns
    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "Vehicle_condition", "multiple_deliveries"]:
        if col in cleaned_df.columns:
            extracted = (
                cleaned_df[col]
                .astype(str)
                .str.extract(r"(\d+\.?\d*)")[0]
            )
            cleaned_df[col] = pd.to_numeric(extracted, errors="coerce")

    # Step 5: Compute or clean distance_km column
    if "distance_km" in cleaned_df.columns:
        extracted = (
            cleaned_df["distance_km"]
            .astype(str)
            .str.extract(r"(\d+\.?\d*)")[0]
        )
        cleaned_df["distance_km"] = pd.to_numeric(extracted, errors="coerce")
    else:
        # Calculate Haversine distance using latitude and longitude columns if present
        lat_cols = [c for c in cleaned_df.columns if "lat" in c.lower()]
        lon_cols = [c for c in cleaned_df.columns if "lon" in c.lower() or "lng" in c.lower()]
        
        rest_lat = [c for c in lat_cols if "rest" in c.lower()]
        rest_lon = [c for c in lon_cols if "rest" in c.lower()]
        del_lat = [c for c in lat_cols if "del" in c.lower() or "loc" in c.lower()]
        del_lon = [c for c in lon_cols if "del" in c.lower() or "loc" in c.lower()]

        if rest_lat and rest_lon and del_lat and del_lon:
            # Convert lat/lon columns to float
            r_lat = pd.to_numeric(cleaned_df[rest_lat[0]], errors="coerce")
            r_lon = pd.to_numeric(cleaned_df[rest_lon[0]], errors="coerce")
            d_lat = pd.to_numeric(cleaned_df[del_lat[0]], errors="coerce")
            d_lon = pd.to_numeric(cleaned_df[del_lon[0]], errors="coerce")

            cleaned_df["distance_km"] = haversine_distance(r_lat, r_lon, d_lat, d_lon)

    # Step 6: Filter out non-positive invalid values safely without wiping valid records
    if "Time_taken (min)" in cleaned_df.columns:
        valid_time_mask = cleaned_df["Time_taken (min)"].notnull() & (cleaned_df["Time_taken (min)"] > 0)
        if valid_time_mask.sum() > 0:
            cleaned_df = cleaned_df[valid_time_mask]

    if "distance_km" in cleaned_df.columns:
        valid_dist_mask = cleaned_df["distance_km"].notnull() & (cleaned_df["distance_km"] >= 0)
        if valid_dist_mask.sum() > 0:
            cleaned_df = cleaned_df[valid_dist_mask]

    # Step 7: Remove exact duplicate records
    cleaned_df = cleaned_df.drop_duplicates()

    return cleaned_df


def clean_dataset_with_report(df):
    """
    Clean the food delivery dataset and return a comprehensive audit report.

    Args:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        tuple: (pd.DataFrame, dict) Cleaned DataFrame and detailed cleaning audit report.
    """
    raw_rows, raw_cols = df.shape
    raw_duplicates = int(df.duplicated().sum())
    missing_before = df.isnull().sum().to_dict()

    # Track column mapping dynamically
    df_temp = df.copy()
    df_temp.columns = df_temp.columns.str.strip()
    canonical_map = {}
    for col in df_temp.columns:
        col_clean = col.strip().lower().replace(" ", "_")
        if "time_taken" in col_clean or "delivery_time" in col_clean or "duration" in col_clean:
            canonical_map[col] = "Time_taken (min)"
        elif "traffic" in col_clean:
            canonical_map[col] = "Road_traffic_density"
        elif "weather" in col_clean:
            canonical_map[col] = "Weather_conditions"
        elif "age" in col_clean:
            canonical_map[col] = "Delivery_person_Age"
        elif "rating" in col_clean:
            canonical_map[col] = "Delivery_person_Ratings"
        elif "distance" in col_clean:
            canonical_map[col] = "distance_km"
        elif "city" in col_clean:
            canonical_map[col] = "City"
        elif "vehicle_condition" in col_clean:
            canonical_map[col] = "Vehicle_condition"

    cleaned_df = clean_dataset(df)

    # Column-level comparison dictionary
    col_audit = {}
    for raw_col in df.columns:
        clean_target = canonical_map.get(raw_col.strip(), raw_col.strip())
        raw_null = int(df[raw_col].isnull().sum())
        clean_null = int(cleaned_df[clean_target].isnull().sum()) if clean_target in cleaned_df.columns else "Dropped"
        col_audit[raw_col] = {
            "mapped_to": clean_target,
            "raw_dtype": str(df[raw_col].dtype),
            "clean_dtype": str(cleaned_df[clean_target].dtype) if clean_target in cleaned_df.columns else "N/A",
            "raw_missing": raw_null,
            "raw_missing_pct": round((raw_null / raw_rows) * 100, 2) if raw_rows > 0 else 0.0,
            "clean_missing": clean_null,
            "clean_missing_pct": round((clean_null / len(cleaned_df)) * 100, 2) if (clean_target in cleaned_df.columns and len(cleaned_df) > 0 and isinstance(clean_null, int)) else 0.0,
        }

    audit_report = {
        "raw_rows": raw_rows,
        "raw_cols": raw_cols,
        "cleaned_rows": len(cleaned_df),
        "cleaned_cols": len(cleaned_df.columns),
        "duplicates_removed": raw_duplicates,
        "missing_before": missing_before,
        "missing_after": cleaned_df.isnull().sum().to_dict(),
        "total_missing_before": sum(missing_before.values()),
        "total_missing_after": int(cleaned_df.isnull().sum().sum()),
        "column_mapping": canonical_map,
        "column_audit": col_audit,
        "dtypes_before": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "dtypes_after": {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()},
    }

    return cleaned_df, audit_report
