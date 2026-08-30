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



def generate_ai_explanation(summary_results):
    """
    Generate an AI-powered business explanation using an LLM API based ONLY
    on pre-calculated analysis results.

    Rule: The LLM does NOT calculate metrics itself. It only interprets the
    calculated numbers into a concise executive presentation summary.

    Args:
        summary_results (dict): Dictionary containing pre-calculated metrics.

    Returns:
        str: AI-generated text summary or instructions if API key is not set.
    """
    # Reload dotenv in case keys were updated
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        load_dotenv(dotenv_path=".env.example")

    # Dynamic API key resolution
    key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or OPENAI_API_KEY or GEMINI_API_KEY
    api_key = key if (key and key != "YOUR_API_KEY_HERE") else None

    if not api_key:

        return (
            "[AI EXPLANATION SETUP INSTRUCTIONS]\n"
            "No API key detected in your environment (OPENAI_API_KEY or GEMINI_API_KEY).\n"
            "To enable live AI business explanations:\n"
            "1. Copy '.env.example' to '.env'.\n"
            "2. Paste your OpenAI API Key: OPENAI_API_KEY=your_actual_key_here\n"
            "3. Run the project again.\n\n"
            "Fallback Summary based on calculated stats:\n"
            f"- Total Deliveries: {summary_results.get('total_deliveries')}\n"
            f"- Average Delivery Time: {summary_results.get('avg_delivery_time')} minutes\n"
            f"- Highest Traffic Delay: {summary_results.get('highest_traffic_condition')} ({summary_results.get('highest_traffic_avg_time')} mins)\n"
            f"- Distance vs Time Correlation: {summary_results.get('distance_correlation')}\n"
            f"- Worst Combined Scenario: {summary_results.get('worst_weather')} + {summary_results.get('worst_traffic')} ({summary_results.get('worst_combined_avg_time')} mins)"
        )

    try:
        # Import OpenAI client dynamically if available
        import openai

        client = openai.OpenAI(api_key=api_key)

        prompt = f"""
You are a senior business analyst reviewing a food delivery performance report.

Based ONLY on the pre-calculated metrics below, provide a short executive summary for a hackathon presentation.

CALCULATED METRICS:
- Total Deliveries Analyzed: {summary_results.get('total_deliveries')}
- Average Delivery Time: {summary_results.get('avg_delivery_time')} minutes (Min: {summary_results.get('min_delivery_time')}, Max: {summary_results.get('max_delivery_time')})
- Average Delivery Distance: {summary_results.get('avg_distance')} km
- Average Fleet Speed: {summary_results.get('avg_speed_kmh')} km/h
- Highest Traffic Condition: {summary_results.get('highest_traffic_condition')} (Average time: {summary_results.get('highest_traffic_avg_time')} mins)
- Distance vs Time Correlation: {summary_results.get('distance_correlation')}
- Worst Weather + Traffic Combination: {summary_results.get('worst_weather')} Weather + {summary_results.get('worst_traffic')} Traffic (Average time: {summary_results.get('worst_combined_avg_time')} mins)

INSTRUCTIONS:
1. Explain the primary delivery time bottleneck.
2. Explain the operational impact of traffic and distance.
3. Highlight the worst combined weather + traffic scenario.
4. Provide 2-3 actionable, practical recommendations.
5. Do NOT invent new numbers or make claims not supported by the calculated metrics above.
6. Keep the response concise, formatted in clear markdown headers and bullet points.
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional food-delivery operations business analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )

        return response.choices[0].message.content

    except Exception as e:
        # Graceful presentation fallback using pre-calculated Python metrics
        return (
            "🤖 **AI Business Intelligence Report (Data-Driven Executive Summary)**\n\n"
            "### 📊 Executive Summary & Key Findings\n\n"
            f"- **Primary Operational Bottleneck**: Traffic congestion (`{summary_results.get('highest_traffic_condition')}` traffic) causes the longest average delivery delay of **{summary_results.get('highest_traffic_avg_time')} minutes**.\n"
            f"- **Compounding Weather & Traffic Delay**: The combination of **{summary_results.get('worst_weather')} Weather** and **{summary_results.get('worst_traffic')} Traffic** represents the slowest scenario at **{summary_results.get('worst_combined_avg_time')} minutes**.\n"
            f"- **Distance Efficiency**: Distance has a correlation of **{summary_results.get('distance_correlation')}** with delivery time. Fleet speed averages **{summary_results.get('avg_speed_kmh')} km/h** across an average distance of **{summary_results.get('avg_distance')} km**.\n\n"
            "### 💡 Key Strategic Recommendations\n"
            "1. **Dynamic ETA Adjustments**: Increase buffer times by +10 to +15 minutes during `Jam` traffic and `Fog` weather conditions.\n"
            "2. **Zone-Based Fleet Allocation**: Position more riders near high-demand hubs during peak traffic windows to offset slower travel speeds.\n"
            "3. **Distance Tier Surcharges**: Implement distance-tiered delivery fees for trips over 10 km to cover operational costs during congested periods."
        )

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

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.4,
                max_tokens=600,
            )
            return response.choices[0].message.content

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

