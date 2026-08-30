import os
import sys
from src.data_loader import load_raw_data, inspect_raw_data
from src.data_cleaning import clean_dataset
from src.analysis import (
    calculate_basic_metrics,
    answer_q1_traffic_impact,
    answer_q2_distance_impact,
    answer_q3_combined_conditions,
    generate_business_insights,
)
from src.visualization import plot_traffic_impact, plot_distance_vs_time, plot_weather_impact
from src.ai_explanation import generate_ai_explanation


def main():
    """
    Main entry point for the Food Delivery Analytics Challenge pipeline.
    """
    print("=" * 50)
    print("      FOOD DELIVERY ANALYTICS CHALLENGE")
    print("=" * 50)
    print()

    # Step 1: Discover and load dataset
    try:
        raw_df, dataset_path = load_raw_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Step 2: Inspect raw dataset
    raw_inspection = inspect_raw_data(raw_df)

    print("DATASET OVERVIEW")
    print("-" * 30)
    print(f"Dataset Path: {dataset_path}")
    print(f"Total Records (Rows): {raw_inspection['num_rows']:,}")
    print(f"Total Columns: {raw_inspection['num_cols']}")
    print(f"Duplicate Records: {raw_inspection['duplicate_count']}")
    print("Missing Values Summary:")
    for col, count in raw_inspection["missing_values"].items():
        if count > 0:
            print(f"  - {col}: {count:,}")
    print()

    # Step 3: Clean the dataset
    print("Cleaning dataset...")
    df = clean_dataset(raw_df)
    print(f"Cleaned Dataset Records: {len(df):,}")
    print()

    # Step 4: Perform basic analysis
    basic_metrics = calculate_basic_metrics(df)

    print("BASIC ANALYSIS")
    print("-" * 30)
    print(f"Total Valid Deliveries: {basic_metrics['total_deliveries']:,}")
    print(f"Average Delivery Time: {basic_metrics['avg_delivery_time']} minutes")
    print(f"Minimum Delivery Time: {basic_metrics['min_delivery_time']} minutes")
    print(f"Maximum Delivery Time: {basic_metrics['max_delivery_time']} minutes")
    print(f"Average Delivery Distance: {basic_metrics['avg_distance']} km")
    print(f"Average Delivery Speed: {basic_metrics['avg_speed_kmh']} km/h")
    print(f"Average Delivery Person Rating: {basic_metrics['avg_rating']} / 5.0")
    print(f"Average Delivery Person Age: {basic_metrics['avg_age']} years")
    print()

    # Step 5: Answer Competition Questions
    print("COMPETITION QUESTIONS")
    print("-" * 30)

    # Q1: Traffic Impact
    q1_res = answer_q1_traffic_impact(df)
    print("Q1 - Traffic Impact:")
    print(f"  Highest average delivery time condition: {q1_res['highest_condition']}")
    print(f"  Average delivery time: {q1_res['highest_avg_time']} minutes")
    print()

    # Q2: Distance Impact
    q2_res = answer_q2_distance_impact(df)
    print("Q2 - Distance Impact:")
    print(f"  Correlation (Distance vs Time): {q2_res['correlation']}")
    print(f"  Conclusion: {q2_res['interpretation']}")
    print("  Average Delivery Time by Distance Range:")
    for b_label, b_time in q2_res["distance_bins_table"].items():
        print(f"    - {b_label}: {b_time} minutes")
    print()

    # Q3: Combined Conditions
    q3_res = answer_q3_combined_conditions(df)
    print("Q3 - Combined Weather & Traffic Conditions:")
    print("  Worst combination with highest average delivery time:")
    print(f"    Weather: {q3_res['worst_weather']}")
    print(f"    Traffic Density: {q3_res['worst_traffic']}")
    print(f"    Average Delivery Time: {q3_res['highest_avg_time']} minutes")
    print()

    # Step 6: Generate Visualizations
    print("GENERATING VISUALIZATIONS")
    print("-" * 30)
    chart1 = plot_traffic_impact(q1_res["traffic_table"])
    chart2 = plot_distance_vs_time(df)
    chart3 = plot_weather_impact(df)

    if chart1:
        print(f"  Saved Chart 1: {chart1}")
    if chart2:
        print(f"  Saved Chart 2: {chart2}")
    if chart3:
        print(f"  Saved Chart 3: {chart3}")
    print()

    # Step 7: Formulate Data-Driven Business Insights
    insights = generate_business_insights(basic_metrics, q1_res, q2_res, q3_res)
    print("DATA-DRIVEN BUSINESS INSIGHTS")
    print("-" * 30)
    for ins in insights:
        print(f"Insight {ins['insight_num']}: {ins['title']}")
        print(f"  - Finding: {ins['finding']}")
        print(f"  - Evidence: {ins['evidence']}")
        print(f"  - Business Meaning: {ins['business_meaning']}")
        print(f"  - Action: {ins['recommended_action']}")
        print()

    # Step 8: AI-Powered Business Explanation
    summary_dict = {
        "total_deliveries": basic_metrics["total_deliveries"],
        "avg_delivery_time": basic_metrics["avg_delivery_time"],
        "min_delivery_time": basic_metrics["min_delivery_time"],
        "max_delivery_time": basic_metrics["max_delivery_time"],
        "avg_distance": basic_metrics["avg_distance"],
        "avg_speed_kmh": basic_metrics["avg_speed_kmh"],
        "highest_traffic_condition": q1_res["highest_condition"],
        "highest_traffic_avg_time": q1_res["highest_avg_time"],
        "distance_correlation": q2_res["correlation"],
        "worst_weather": q3_res["worst_weather"],
        "worst_traffic": q3_res["worst_traffic"],
        "worst_combined_avg_time": q3_res["highest_avg_time"],
    }

    print("AI BUSINESS EXPLANATION")
    print("-" * 30)
    ai_summary = generate_ai_explanation(summary_dict)
    print(ai_summary)
    print()
    print("=" * 50)
    print("ANALYSIS PIPELINE COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
