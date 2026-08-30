import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Optional Seaborn styling for sleek aesthetics
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def ensure_output_directory(output_dir="outputs/charts"):
    """
    Ensure that the output directory for saving charts exists.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def plot_traffic_impact(traffic_table, output_dir="outputs/charts"):
    """
    Visualization 1: Bar chart showing Average Delivery Time by Traffic Density.

    Args:
        traffic_table (dict): Dictionary mapping traffic density categories to average delivery time.
        output_dir (str): Folder path to save the generated chart.

    Returns:
        str: Saved image file path.
    """
    ensure_output_directory(output_dir)

    if not traffic_table:
        print("No traffic data available for plotting.")
        return None

    # Prepare data
    traffic_conditions = list(traffic_table.keys())
    avg_times = list(traffic_table.values())

    # Create figure
    plt.figure(figsize=(9, 5.5))
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"] if len(traffic_conditions) == 4 else "#3498db"
    bars = plt.bar(traffic_conditions, avg_times, color=colors, edgecolor="#2c3e50", width=0.55, linewidth=1.2)

    # Add chart title and axis labels
    plt.title("Average Delivery Time by Traffic Density", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Road Traffic Density", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Average Delivery Time (minutes)", fontsize=12, fontweight="bold", labelpad=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim(0, max(avg_times) * 1.18 if avg_times else 40)

    # Add numeric annotations on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.6,
            f"{height:.2f} min",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#2c3e50"
        )

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "traffic_impact.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    return chart_path


def plot_distance_vs_time(df, output_dir="outputs/charts"):
    """
    Visualization 2: Scatter plot showing Delivery Distance vs Delivery Time.

    Args:
        df (pd.DataFrame): Cleaned food delivery DataFrame.
        output_dir (str): Folder path to save the generated chart.

    Returns:
        str: Saved image file path.
    """
    ensure_output_directory(output_dir)

    if "distance_km" not in df.columns or "Time_taken (min)" not in df.columns:
        print("Missing required columns for distance vs time scatter plot.")
        return None

    # Filter out NaNs and non-positive values
    clean_scatter = df[(df["distance_km"] > 0) & (df["Time_taken (min)"] > 0)].dropna(subset=["distance_km", "Time_taken (min)"])
    if len(clean_scatter) == 0:
        return None

    # Subsample if dataset is large for crisp rendering
    sample_df = clean_scatter.sample(min(3000, len(clean_scatter)), random_state=42)

    # Create figure and scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(
        sample_df["distance_km"],
        sample_df["Time_taken (min)"],
        alpha=0.35,
        c="#2563eb",
        edgecolors="none",
        s=30,
        label="Delivery Orders"
    )

    # Add linear trendline
    try:
        z = np.polyfit(sample_df["distance_km"], sample_df["Time_taken (min)"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(sample_df["distance_km"].min(), sample_df["distance_km"].max(), 100)
        plt.plot(x_range, p(x_range), color="#dc2626", linewidth=2.5, linestyle="--", label="OLS Trendline")
    except Exception:
        pass

    # Add title and labels
    plt.title("Delivery Distance (km) vs. Delivery Time (minutes)", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Delivery Distance (km)", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Delivery Time (minutes)", fontsize=12, fontweight="bold", labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "distance_vs_time.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    return chart_path


def plot_weather_impact(df, output_dir="outputs/charts"):
    """
    Visualization 3: Bar chart showing Average Delivery Time by Weather Condition.

    Args:
        df (pd.DataFrame): Cleaned food delivery DataFrame.
        output_dir (str): Folder path to save the generated chart.

    Returns:
        str: Saved image file path.
    """
    ensure_output_directory(output_dir)

    if "Weather_conditions" not in df.columns or "Time_taken (min)" not in df.columns:
        return None

    weather_avg = (
        df.groupby("Weather_conditions")["Time_taken (min)"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .dropna()
    )

    if len(weather_avg) == 0:
        return None

    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(weather_avg.index, weather_avg.values, color="#10b981", edgecolor="#047857", width=0.55, linewidth=1.2)

    plt.title("Average Delivery Time by Weather Condition", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Weather Condition", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Average Delivery Time (minutes)", fontsize=12, fontweight="bold", labelpad=10)
    plt.xticks(rotation=15, fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim(0, max(weather_avg.values) * 1.18)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.6,
            f"{height:.2f} min",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#064e3b"
        )

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "weather_impact.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    return chart_path
