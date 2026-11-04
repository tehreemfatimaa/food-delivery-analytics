import os
from dotenv import load_dotenv

# Load environment variables from .env file or .env.example fallback
load_dotenv()
if not os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    load_dotenv(dotenv_path=".env.example")

# ==============================================================================
# API KEY CONFIGURATION PLACEHOLDERS
# You can set your API key in the environment OR replace "YOUR_API_KEY_HERE" below.
# ==============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY_HERE"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "YOUR_API_KEY_HERE"



def build_executive_briefing_report(summary_results):
    """
    Construct a comprehensive, data-grounded, publication-grade executive
    briefing report based on pre-calculated metrics.
    """
    total_del = summary_results.get("total_deliveries", 0)
    total_str = f"{total_del:,}" if isinstance(total_del, (int, float)) else str(total_del)
    avg_time = summary_results.get("avg_delivery_time", "N/A")
    min_time = summary_results.get("min_delivery_time", "N/A")
    max_time = summary_results.get("max_delivery_time", "N/A")
    avg_dist = summary_results.get("avg_distance", "N/A")
    avg_speed = summary_results.get("avg_speed_kmh", "N/A")
    avg_rating = summary_results.get("avg_rating", "N/A")
    avg_age = summary_results.get("avg_age", "N/A")
    hi_traffic = summary_results.get("highest_traffic_condition", "N/A")
    hi_traffic_time = summary_results.get("highest_traffic_avg_time", "N/A")
    corr = summary_results.get("distance_correlation", "N/A")
    worst_w = summary_results.get("worst_weather", "N/A")
    worst_t = summary_results.get("worst_traffic", "N/A")
    worst_comb_time = summary_results.get("worst_combined_avg_time", "N/A")

    # Scope representation
    city = summary_results.get("selected_city", "All Cities")
    traffic = summary_results.get("selected_traffic", "All Densities")
    weather = summary_results.get("selected_weather", "All Weather")
    vehicle = summary_results.get("selected_vehicle", "All Conditions")

    scope_parts = []
    if city and city != "All Cities":
        scope_parts.append(f"City: **{city}**")
    if traffic and traffic != "All Densities":
        scope_parts.append(f"Traffic: **{traffic}**")
    if weather and weather != "All Weather":
        scope_parts.append(f"Weather: **{weather}**")
    if vehicle and vehicle != "All Conditions":
        scope_parts.append(f"Vehicle Rating: **{vehicle}**")

    scope_str = " | ".join(scope_parts) if scope_parts else "Full Fleet Network (Unfiltered)"

    report = f"""# 📋 SWIFTPULSE EXECUTIVE OPERATIONS BRIEFING
**Operational Intelligence Report** | **Active Scope:** {scope_str} | **Verified Records:** {total_str} trips

---

### 🎯 1. Executive Summary & Operational Health
Deliveries evaluated under the active cohort average **{avg_time} minutes** per order across a mean fulfillment radius of **{avg_dist} km**, yielding an operational transit velocity of **{avg_speed} km/h**. 

Statistical diagnostic telemetry identifies **{hi_traffic} traffic density** combined with **{worst_w} weather** as the principal drivers of order latency and fulfillment variance across the fleet.

| Operational Parameter | Observed Metric | Operational Benchmark / Impact |
| :--- | :--- | :--- |
| **Active Cohort Volume** | `{total_str}` trips | Verified delivery records |
| **Mean Order Duration** | `{avg_time} min` | Observed range: `{min_time} – {max_time} min` |
| **Average Delivery Distance** | `{avg_dist} km` | Great-circle Haversine computation |
| **Fleet Transit Velocity** | `{avg_speed} km/h` | Effective road speed under current conditions |
| **Primary Traffic Bottleneck** | `{hi_traffic}` | Causes average delay of `{hi_traffic_time} min` |
| **Peak Multi-Hazard Delay** | `{worst_w} + {worst_t}` | Cascades to `{worst_comb_time} min` peak duration |
| **Courier Performance Profile** | `{avg_rating} / 5.0` | Average courier age: `{avg_age} years` |

---

### 🔍 2. In-Depth Operational Diagnostics

#### 🚦 Road Traffic & Congestion Drag
Deliveries encountering **`{hi_traffic}` traffic** experience an average fulfillment duration of **`{hi_traffic_time} minutes`**. In dense congestion corridors, static dispatch models consistently fail to anticipate localized bottlenecks, causing couriers to sit idle in traffic, burning fuel and extending delivery windows beyond initial customer estimates.

#### 🌧️ Environmental & Multi-Hazard Delay Cascades
Adverse atmospheric conditions compound transit delays. The compound interaction of **`{worst_w}` weather** and **`{worst_t}` traffic** represents the single slowest operational state in the fleet, reaching an average of **`{worst_comb_time} minutes`**. Reduced road traction, impaired rider visibility, and weather-related kitchen dispatch bottlenecks create compounding wait times.

#### 📏 Spatial Distance & Transit Velocity Dynamics
Delivery distance exhibits a Pearson correlation coefficient of **`{corr}`** with delivery duration. With average transit velocity recorded at **`{avg_speed} km/h`**, fulfillment duration scales with mileage. Long-range deliveries (over 10 km) place disproportionate strain on rider availability, making them vulnerable to severe delay spikes when intersecting congested routes.

#### 🛵 Courier Cohort Reliability
The active delivery partner pool maintains a strong average rating of **`{avg_rating} / 5.0`** across an average courier age of **`{avg_age} years`**. This customer satisfaction rating demonstrates that courier professionalism remains resilient even during demanding operational conditions.

---

### 💡 3. Prioritized Strategic Action Plan

#### 🔴 Priority 1: Immediate Dispatch & ETA Calibration (0–48 Hours)
* **Dynamic Buffer Extensions**: Automatically inject an adaptive **+8 to +15 minute buffer** into customer Estimated Time of Arrival (ETA) predictions whenever `{hi_traffic}` traffic or `{worst_w}` weather conditions are detected to protect customer satisfaction.
* **Proactive Status Transparency**: Display real-time weather and traffic delay notices in the customer app to reduce cancelation rates and support inquiries.

#### 🟡 Priority 2: Fleet Staging & Zonal Balancing (1–2 Weeks)
* **Hub-Centric Pre-Positioning**: Re-allocate 15–20% more couriers to dense merchant clusters prior to peak traffic windows to shorten pickup transit times.
* **Inclement Weather Radius Management**: Dynamically contract maximum delivery radiuses to **7.0–8.0 km** during extreme weather events (`{worst_w}`) to keep couriers concentrated in high-density corridors.

#### 🟢 Priority 3: Commercial Policy & Unit Economics (Ongoing)
* **Distance & Hazard Tier Surcharges**: Institute incremental delivery fees for long-range trips (10+ km) and severe weather conditions to fund enhanced courier safety incentives and protect unit margins.
* **Micro-Fulfillment Expansion**: Identify merchant corridors with high long-distance order demand for cloud kitchen and micro-hub partnerships.
"""
    return report


def generate_ai_explanation(summary_results):
    """
    Generate an AI-powered business explanation using an LLM API based ONLY
    on pre-calculated analysis results. Falls back gracefully to an
    expert-crafted operational briefing report.

    Rule: The LLM does NOT calculate metrics itself. It only interprets the
    calculated numbers into a concise executive presentation summary.

    Args:
        summary_results (dict): Dictionary containing pre-calculated metrics.

    Returns:
        str: AI-generated text summary or structured executive report.
    """
    # Reload dotenv in case keys were updated
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        load_dotenv(dotenv_path=".env.example")

    # Dynamic API key resolution
    key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or OPENAI_API_KEY or GEMINI_API_KEY
    api_key = key if (key and key != "YOUR_API_KEY_HERE") else None

    if not api_key:
        return build_executive_briefing_report(summary_results)

    try:
        # Import OpenAI client dynamically if available
        import openai

        client = openai.OpenAI(api_key=api_key)

        prompt = f"""
You are an executive operations consultant reviewing a high-resolution food delivery telemetry report.

Based ONLY on the pre-calculated metrics below, generate a formal, high-impact Executive Operations Briefing.

CALCULATED METRICS:
- Scope: City={summary_results.get('selected_city', 'All')}, Traffic={summary_results.get('selected_traffic', 'All')}, Weather={summary_results.get('selected_weather', 'All')}, Vehicle={summary_results.get('selected_vehicle', 'All')}
- Total Deliveries Analyzed: {summary_results.get('total_deliveries')}
- Average Delivery Time: {summary_results.get('avg_delivery_time')} minutes (Min: {summary_results.get('min_delivery_time')}, Max: {summary_results.get('max_delivery_time')})
- Average Delivery Distance: {summary_results.get('avg_distance')} km
- Average Fleet Speed: {summary_results.get('avg_speed_kmh')} km/h
- Delivery Driver Rating: {summary_results.get('avg_rating')} / 5.0 (Average Age: {summary_results.get('avg_age')} years)
- Highest Traffic Condition: {summary_results.get('highest_traffic_condition')} (Average time: {summary_results.get('highest_traffic_avg_time')} mins)
- Distance vs Time Correlation: {summary_results.get('distance_correlation')}
- Worst Weather + Traffic Combination: {summary_results.get('worst_weather')} Weather + {summary_results.get('worst_traffic')} Traffic (Average time: {summary_results.get('worst_combined_avg_time')} mins)

STRUCTURE YOUR REPORT AS FOLLOWS:
# 📋 SWIFTPULSE EXECUTIVE OPERATIONS BRIEFING
1. 🎯 Executive Summary & Operational Health (with markdown KPI overview table)
2. 🔍 In-Depth Operational Diagnostics (Traffic Drag, Weather Cascades, Distance Dynamics, Courier Health)
3. 💡 Prioritized Strategic Action Plan (Priority 1: Immediate 0-48h, Priority 2: 1-2 Weeks, Priority 3: Long-Term)

INSTRUCTIONS:
- Use clean GitHub-flavored markdown with clean headers, bullet points, and tables.
- Do NOT invent numbers; strictly use the metrics provided above.
- Ensure the tone is authoritative, strategic, and polished for executive review.
"""

        # Try modern models first, falling back if needed
        model_name = "gpt-4o-mini"
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a professional logistics operations director and data analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=900,
            )
            return response.choices[0].message.content
        except Exception:
            # Fall back to structured executive report if model fails or quota is exhausted
            return build_executive_briefing_report(summary_results)

    except Exception:
        # Graceful presentation fallback using pre-calculated Python metrics
        return build_executive_briefing_report(summary_results)

def answer_user_question_ai(user_query, summary_results, conversation_history=None):
    """
    Answer user questions about the food delivery analytics using an LLM API or
    a data-grounded fallback assistant based on pre-calculated metrics.

    Args:
        user_query (str): The question asked by the user.
        summary_results (dict): Dictionary of pre-calculated metrics.
        conversation_history (list, optional): Previous messages for context.

    Returns:
        str: AI-generated or fallback response to the user's question.
    """
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        load_dotenv(dotenv_path=".env.example")

    key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or OPENAI_API_KEY or GEMINI_API_KEY
    api_key = key if (key and key != "YOUR_API_KEY_HERE") else None

    # Format ground-truth metrics context
    metrics_context = f"""
PROJECT METRICS CONTEXT:
- Total Deliveries Analyzed: {summary_results.get('total_deliveries', 'N/A')}
- Average Delivery Time: {summary_results.get('avg_delivery_time', 'N/A')} minutes (Min: {summary_results.get('min_delivery_time', 'N/A')}, Max: {summary_results.get('max_delivery_time', 'N/A')})
- Average Delivery Distance: {summary_results.get('avg_distance', 'N/A')} km
- Average Fleet Speed: {summary_results.get('avg_speed_kmh', 'N/A')} km/h
- Average Delivery Driver Rating: {summary_results.get('avg_rating', 'N/A')} / 5.0
- Average Delivery Driver Age: {summary_results.get('avg_age', 'N/A')} years
- Highest Traffic Delay Condition: {summary_results.get('highest_traffic_condition', 'N/A')} (Avg time: {summary_results.get('highest_traffic_avg_time', 'N/A')} mins)
- Distance vs Time Correlation: {summary_results.get('distance_correlation', 'N/A')}
- Worst Combined Scenario: {summary_results.get('worst_weather', 'N/A')} Weather + {summary_results.get('worst_traffic', 'N/A')} Traffic (Avg time: {summary_results.get('worst_combined_avg_time', 'N/A')} mins)
"""

    if api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Food Delivery AI Assistant, an expert analytics consultant reviewing food delivery operational performance. "
                        "Answer the user's questions clearly, accurately, and professionally based on the provided ground-truth dataset metrics context below. "
                        "Use clean GitHub-style markdown, bullet points, and tables where appropriate.\n"
                        f"{metrics_context}"
                    )
                }
            ]

            if conversation_history:
                for msg in conversation_history[-6:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({"role": "user", "content": user_query})

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.4,
                    max_tokens=600,
                )
                return response.choices[0].message.content
            except Exception:
                pass

        except Exception as e:
            err_msg = str(e)
            # Fall through to grounded rule-based answer below if API fails or quota runs out

    # Rule-Based Data Assistant Fallback (when no API key or API limit reached)
    q_lower = user_query.lower()

    if any(k in q_lower for k in ["traffic", "jam", "congestion", "delay"]):
        return (
            f"### 🚦 Traffic Analysis Answer\n\n"
            f"- **Highest Traffic Delay Condition**: `{summary_results.get('highest_traffic_condition', 'N/A')}`\n"
            f"- **Average Time under Worst Traffic**: **{summary_results.get('highest_traffic_avg_time', 'N/A')} minutes**\n"
            f"- **Overall Fleet Average Time**: **{summary_results.get('avg_delivery_time', 'N/A')} minutes**\n\n"
            f"**Key Takeaway**: Deliveries during `{summary_results.get('highest_traffic_condition', 'N/A')}` traffic experience the longest duration. "
            f"We recommend increasing Estimated Time of Arrival (ETA) buffers during peak traffic windows."
        )

    elif any(k in q_lower for k in ["weather", "storm", "fog", "rain", "combined", "worst"]):
        return (
            f"### 🌧️ Weather & Combined Bottlenecks\n\n"
            f"- **Worst Operational Scenario**: **{summary_results.get('worst_weather', 'N/A')} Weather + {summary_results.get('worst_traffic', 'N/A')} Traffic**\n"
            f"- **Average Delivery Duration**: **{summary_results.get('worst_combined_avg_time', 'N/A')} minutes**\n\n"
            f"**Operational Insight**: Adverse weather combined with traffic congestion creates a compounding delay effect across the fleet."
        )

    elif any(k in q_lower for k in ["distance", "speed", "far", "km", "correlation"]):
        return (
            f"### 📏 Distance & Speed Analytics\n\n"
            f"- **Average Delivery Distance**: **{summary_results.get('avg_distance', 'N/A')} km**\n"
            f"- **Average Fleet Speed**: **{summary_results.get('avg_speed_kmh', 'N/A')} km/h**\n"
            f"- **Distance vs. Time Correlation**: **{summary_results.get('distance_correlation', 'N/A')}**\n\n"
            f"**Insight**: Distance positively correlates with delivery time, but urban traffic caps overall speed efficiency."
        )

    elif any(k in q_lower for k in ["recommend", "action", "improve", "solution", "strategy"]):
        return (
            "### 💡 Strategic Operational Recommendations\n\n"
            "1. **Dynamic ETA Adjustments**: Add +10 to +15 minute buffer times for orders placed during `Jam` traffic and `Fog`/`Stormy` weather.\n"
            "2. **Zonal Rider Dispatching**: Station higher concentrations of delivery partners near high-order density hubs during rush hours.\n"
            "3. **Distance Tier Pricing**: Implement tiered delivery fees for orders over 10 km to compensate riders and optimize fulfillment radiuses."
        )

    elif any(k in q_lower for k in ["rating", "driver", "age", "rider"]):
        return (
            f"### 👨‍🛵 Delivery Partner Metrics\n\n"
            f"- **Average Delivery Driver Rating**: **{summary_results.get('avg_rating', 'N/A')} / 5.0**\n"
            f"- **Average Delivery Driver Age**: **{summary_results.get('avg_age', 'N/A')} years**\n"
            f"- **Total Deliveries Analyzed**: **{summary_results.get('total_deliveries', 'N/A'):,}**"
        )

    else:
        return (
            f"### 📊 Food Delivery Performance Summary\n\n"
            f"Here are the core calculated metrics for your query:\n"
            f"- **Total Deliveries Analyzed**: {summary_results.get('total_deliveries', 'N/A'):,}\n"
            f"- **Average Delivery Time**: {summary_results.get('avg_delivery_time', 'N/A')} minutes\n"
            f"- **Average Distance**: {summary_results.get('avg_distance', 'N/A')} km\n"
            f"- **Average Speed**: {summary_results.get('avg_speed_kmh', 'N/A')} km/h\n"
            f"- **Highest Traffic Delay**: {summary_results.get('highest_traffic_condition', 'N/A')} ({summary_results.get('highest_traffic_avg_time', 'N/A')} mins)\n"
            f"- **Worst Scenario**: {summary_results.get('worst_weather', 'N/A')} Weather + {summary_results.get('worst_traffic', 'N/A')} Traffic ({summary_results.get('worst_combined_avg_time', 'N/A')} mins)\n\n"
            f"*Tip: You can ask me specific questions about traffic delays, weather impact, distance correlation, or strategic recommendations!*"
        )

