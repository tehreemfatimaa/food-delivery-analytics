import pandas as pd
import numpy as np


def calculate_basic_metrics(df):
    """
    Calculate summary metrics for the dataset.

    Returns:
        dict: Basic statistics including delivery times, distances, speed, rating, and age.
    """
    metrics = {
        "total_deliveries": len(df),
        "avg_delivery_time": "N/A",
        "min_delivery_time": "N/A",
        "max_delivery_time": "N/A",
        "avg_distance": "N/A",
        "avg_speed_kmh": "N/A",
        "avg_rating": "N/A",
        "avg_age": "N/A",
    }

    if len(df) == 0:
        return metrics

    # Delivery time statistics (minutes) using actual column "Time_taken (min)"
    if "Time_taken (min)" in df.columns:
        valid_times = pd.to_numeric(df["Time_taken (min)"], errors="coerce").dropna()
        if len(valid_times) > 0:
            metrics["avg_delivery_time"] = round(float(valid_times.mean()), 2)
            metrics["min_delivery_time"] = round(float(valid_times.min()), 2)
            metrics["max_delivery_time"] = round(float(valid_times.max()), 2)

    # Delivery distance statistics (km) using actual column "distance_km"
    if "distance_km" in df.columns:
        valid_dist = pd.to_numeric(df["distance_km"], errors="coerce").dropna()
        if len(valid_dist) > 0:
            metrics["avg_distance"] = round(float(valid_dist.mean()), 2)

    # Delivery speed calculation (km/h)
    # Speed (km/h) = distance_km / (Time_taken (min) / 60)
    if "distance_km" in df.columns and "Time_taken (min)" in df.columns:
        # Calculate speed for each row where time > 0 and distance > 0
        dist_num = pd.to_numeric(df["distance_km"], errors="coerce")
        time_num = pd.to_numeric(df["Time_taken (min)"], errors="coerce")
        valid_mask = (time_num > 0) & (dist_num > 0)
        if valid_mask.sum() > 0:
            speeds = dist_num[valid_mask] / (time_num[valid_mask] / 60.0)
            metrics["avg_speed_kmh"] = round(float(speeds.mean()), 2)

    # Delivery-person rating statistics using actual column "Delivery_person_Ratings"
    if "Delivery_person_Ratings" in df.columns:
        valid_ratings = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce").dropna()
        if len(valid_ratings) > 0:
            metrics["avg_rating"] = round(float(valid_ratings.mean()), 2)

    # Delivery-person age statistics using actual column "Delivery_person_Age"
    if "Delivery_person_Age" in df.columns:
        valid_ages = pd.to_numeric(df["Delivery_person_Age"], errors="coerce").dropna()
        if len(valid_ages) > 0:
            metrics["avg_age"] = round(float(valid_ages.mean()), 2)

    return metrics


def answer_q1_traffic_impact(df):
    """
    Competition Question 1:
    Which road traffic condition has the highest average delivery time?

    Returns:
        dict: Traffic condition analysis including highest condition and average time table.
    """
    if "Road_traffic_density" not in df.columns or "Time_taken (min)" not in df.columns:
        return {
            "traffic_table": {},
            "highest_condition": "N/A",
            "highest_avg_time": 0.0,
        }

    # Group deliveries by traffic condition and compute mean delivery time
    traffic_analysis = (
        df.groupby("Road_traffic_density")["Time_taken (min)"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .dropna()  # drop groups where all times are NaN
    )

    # Guard: return safe fallback if no data remains after filtering
    if traffic_analysis.empty:
        return {
            "traffic_table": {},
            "highest_condition": "N/A",
            "highest_avg_time": 0.0,
        }

    highest_condition = traffic_analysis.idxmax()
    highest_avg_time = traffic_analysis.max()

    return {
        "traffic_table": traffic_analysis.to_dict(),
        "highest_condition": highest_condition,
        "highest_avg_time": float(highest_avg_time),
    }


def answer_q2_distance_impact(df):
    """
    Competition Question 2:
    How does delivery distance affect delivery time?

    Returns:
        dict: Correlation value, distance bin table, and relationship interpretation.
    """
    if "distance_km" not in df.columns or "Time_taken (min)" not in df.columns:
        return {
            "correlation": 0.0,
            "interpretation": "N/A",
            "distance_bins_table": {},
        }

    # Guard: require at least 2 valid rows to calculate a meaningful correlation
    valid_df = df[["distance_km", "Time_taken (min)"]].dropna()
    if len(valid_df) < 2:
        return {
            "correlation": 0.0,
            "interpretation": "Not enough data to compute correlation after current filters.",
            "distance_bins_table": {},
        }

    # Calculate Pearson correlation between distance_km and Time_taken (min)
    correlation = round(float(valid_df["distance_km"].corr(valid_df["Time_taken (min)"])), 4)

    # Handle NaN correlation (all values identical edge case)
    if pd.isna(correlation):
        correlation = 0.0

    # Interpret correlation strength and direction
    abs_corr = abs(correlation)
    if abs_corr < 0.2:
        strength = "very weak"
    elif abs_corr < 0.4:
        strength = "weak"
    elif abs_corr < 0.6:
        strength = "moderate"
    elif abs_corr < 0.8:
        strength = "strong"
    else:
        strength = "very strong"

    direction = "increases" if correlation > 0 else "decreases"
    interpretation = f"As delivery distance increases, delivery time generally {direction} (Correlation: {correlation}, indicating a {strength} positive relationship)."

    # Define suitable distance bins based on dataset distribution
    max_dist = df["distance_km"].max()
    bins = [0, 5, 10, 15, max(20, max_dist + 1)]
    labels = ["0-5 km", "5-10 km", "10-15 km", "15+ km"]

    # Bin deliveries into distance ranges and calculate average delivery time per range
    df_temp = df.copy()
    df_temp["Distance_Bin"] = pd.cut(df_temp["distance_km"], bins=bins, labels=labels, right=False)

    distance_bin_avg = (
        df_temp.groupby("Distance_Bin", observed=False)["Time_taken (min)"]
        .mean()
        .round(2)
        .to_dict()
    )

    return {
        "correlation": correlation,
        "interpretation": interpretation,
        "distance_bins_table": distance_bin_avg,
    }


def answer_q3_combined_conditions(df):
    """
    Competition Question 3:
    Which combination of weather condition and traffic density has the highest average delivery time?

    Returns:
        dict: Highest combination details and full table.
    """
    if "Weather_conditions" not in df.columns or "Road_traffic_density" not in df.columns or "Time_taken (min)" not in df.columns:
        return {
            "worst_weather": "N/A",
            "worst_traffic": "N/A",
            "highest_avg_time": 0.0,
            "combined_table": {},
        }

    # Group by Weather_conditions and Road_traffic_density and compute mean delivery time
    combined_analysis = (
        df.groupby(["Weather_conditions", "Road_traffic_density"])["Time_taken (min)"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .dropna()  # drop groups where all times are NaN
    )

    # Guard: return safe fallback if no data remains after filtering
    if combined_analysis.empty:
        return {
            "worst_weather": "N/A",
            "worst_traffic": "N/A",
            "highest_avg_time": 0.0,
            "combined_table": {},
        }

    worst_combination = combined_analysis.idxmax()  # Tuple of (Weather_conditions, Road_traffic_density)
    highest_avg_time = combined_analysis.max()

    # Convert tuple keys to readable strings for dictionary return
    table_dict = {f"{w} + {t}": float(v) for (w, t), v in combined_analysis.items()}

    return {
        "worst_weather": worst_combination[0],
        "worst_traffic": worst_combination[1],
        "highest_avg_time": float(highest_avg_time),
        "combined_table": table_dict,
    }


def generate_business_insights(basic_metrics, q1_res, q2_res, q3_res):
    """
    Generate three data-driven business insights based on calculated data.

    Returns:
        list: List of 3 insight dictionaries containing Finding, Evidence, Business Meaning, and Recommended Action.
    """
    if not basic_metrics or basic_metrics.get("total_deliveries", 0) == 0 or q1_res.get("highest_condition") == "N/A":
        return []

    insights = []

    # Insight 1: Traffic Impact
    insights.append({
        "insight_num": 1,
        "title": "Severe Traffic Delays",
        "finding": f"Deliveries under '{q1_res['highest_condition']}' traffic experience the longest delivery times.",
        "evidence": f"Average delivery time during '{q1_res['highest_condition']}' traffic is {q1_res['highest_avg_time']} minutes, compared to an overall average of {basic_metrics['avg_delivery_time']} minutes.",
        "business_meaning": "Congested traffic significantly impairs fulfillment speeds, risking customer satisfaction and SLA breaches.",
        "recommended_action": "Implement dynamic estimated delivery times (EDTs) and allocate higher delivery partner incentives during peak traffic hours."
    })

    # Insight 2: Combined Extreme Weather & Traffic Impact
    insights.append({
        "insight_num": 2,
        "title": "Compounding Weather and Traffic Bottlenecks",
        "finding": f"The combination of '{q3_res['worst_weather']}' weather and '{q3_res['worst_traffic']}' traffic represents the single slowest operational scenario.",
        "evidence": f"Orders placed under '{q3_res['worst_weather']}' weather + '{q3_res['worst_traffic']}' traffic take an average of {q3_res['highest_avg_time']} minutes to deliver.",
        "business_meaning": "Adverse weather coupled with high traffic creates severe bottleneck cascades across the delivery fleet.",
        "recommended_action": "Introduce buffer time warnings for customers ordering during bad weather and restricted delivery radiuses to maintain reliability."
    })

    # Insight 3: Distance and Speed Efficiency
    insights.append({
        "insight_num": 3,
        "title": "Distance Correlation and Fleet Speed Optimization",
        "finding": f"Delivery distance has a correlation of {q2_res['correlation']} with total delivery time, with average fleet speed recorded at {basic_metrics['avg_speed_kmh']} km/h.",
        "evidence": f"Average distance is {basic_metrics['avg_distance']} km. Deliveries in the 15+ km distance bin take longer on average than short 0-5 km deliveries.",
        "business_meaning": "Distance directly drives delivery duration, but urban congestion limits fleet speed, making long-distance orders disproportionately expensive.",
        "recommended_action": "Optimize delivery radius dispatching and consider micro-fulfillment hubs / cloud kitchens for popular items."
    })

    return insights
