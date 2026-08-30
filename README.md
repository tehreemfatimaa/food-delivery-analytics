# Food Delivery Analytics Challenge

A Python & Data Analysis hackathon project built to analyze food delivery operational data, programmatically answer competition questions regarding delivery delays, generate publication-grade visualizations, and synthesize executive insights using an LLM API.

---

## 1. Project Overview

The objective of this project is to analyze a food-delivery dataset (~38,964 records, 22 columns) to extract actionable insights that optimize delivery performance, reduce wait times, and improve rider allocation. 

**Key Features:**
- **No Hard-coding**: All metrics and competition answers are calculated programmatically from the dataset.
- **Strict Data Science Hierarchy**: Python/Pandas performs 100% of data calculations. The LLM API is strictly used to explain calculated results.
- **Secure API Key Handling**: Uses environment variables (`.env`) without exposing API keys.

---

## 2. Technologies Used

- **Language**: Python 3.9+
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **AI Explanation**: OpenAI / Gemini LLM API (via `python-dotenv` and `openai`)

---

## 3. Project Structure

```text
food_delivery/
├── data/                      # Place your dataset (e.g. food_delivery.csv) here
│   └── .gitkeep
├── src/                       # Modular source code
│   ├── __init__.py
│   ├── data_loader.py         # Dataset discovery & raw metadata inspection
│   ├── data_cleaning.py       # Column standardization, distance math, NaN & duplicate handling
│   ├── analysis.py            # Summary stats, Q1, Q2, Q3 calculations, & business insights
│   ├── visualization.py       # Matplotlib chart generation
│   └── ai_explanation.py      # LLM integration using calculated stats only
├── outputs/                   # Generated output artifacts
│   ├── charts/                # Saved PNG charts (traffic_impact.png, distance_vs_time.png)
│   └── results/               # Results storage
├── app.py                     # Interactive Streamlit Web Dashboard
├── main.py                    # Main CLI pipeline entry point
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── .env.example               # Environment configuration template
└── .gitignore                 # Security & version control rules
```

---

## 4. Dataset Column Reference

The dataset contains approximately **38,964 rows** and **22 columns**. Key columns used in analysis:

| Column | Description |
|---|---|
| `Road_traffic_density` | Traffic density category: `Jam`, `High`, `Medium`, `Low` |
| `Weather_conditions` | Weather at delivery time: `Sunny`, `Cloudy`, `Fog`, `Stormy`, `Sandstorms`, `Windy` |
| `Vehicle_condition` | Numeric vehicle condition rating: `0`, `1`, `2` |
| `Time_taken (min)` | Delivery duration in minutes (the primary delivery-time column) |
| `distance_km` | Delivery distance in kilometers |
| `City` | City category: e.g., `Metropolitian`, `Urban`, `Semi-Urban` |
| `Delivery_person_Age` | Age of the delivery person |
| `Delivery_person_Ratings` | Rating of the delivery person (out of 5) |

---

## 5. Data Cleaning Strategy

The dataset is cleaned programmatically in `src/data_cleaning.py`:
1. **Column Standardization**: Dynamically maps Kaggle header variations to actual dataset column names using fuzzy matching (e.g., `Time_taken(min)` → `Time_taken (min)`, `Weatherconditions` → `Weather_conditions`).
2. **Text Normalization**: Strips whitespace, removes `'conditions '` prefix in `Weather_conditions`, and converts literal `'nan'` strings to `np.nan`.
3. **Numeric Extraction**: Extracts float values from string-formatted fields using regex `r'(\d+\.?\d*)'` (e.g., `(min) 24` → `24.0`) for `Time_taken (min)`, `Delivery_person_Age`, and `Delivery_person_Ratings`.
4. **Coordinate Distance Calculation**: If `distance_km` is absent, calculates Haversine distance in kilometers from restaurant and delivery location coordinates.
5. **Invalid Value Filtering**: Removes rows with non-positive delivery times only when valid rows remain.
6. **Duplicate Removal**: Identifies and drops exact duplicate rows.

> **Important**: Missing values in `Delivery_person_Age`, `Delivery_person_Ratings`, `Time_Orderd`, and `Time_Order_picked` are **not** replaced with zero. They remain `NaN` or are handled with appropriate strategies.

---

## 6. Competition Questions & Methodology

### Q1 — Traffic Impact
- **Question**: *Which road traffic condition has the highest average delivery time?*
- **Method**: Groups data by `Road_traffic_density`, calculates mean `Time_taken (min)`. The highest average time and corresponding condition are extracted dynamically using `.idxmax()` and `.max()`.

### Q2 — Distance Impact
- **Question**: *How does delivery distance affect delivery time?*
- **Method**: Calculates Pearson correlation between `distance_km` and `Time_taken (min)`. Bins distance into ranges (`0-5 km`, `5-10 km`, `10-15 km`, `15+ km`) and calculates average delivery duration per range.

### Q3 — Combined Weather & Traffic Conditions
- **Question**: *Which combination of weather condition and traffic density has the highest average delivery time?*
- **Method**: Groups dataset by `["Weather_conditions", "Road_traffic_density"]`, calculates mean `Time_taken (min)`, and isolates the combination producing the maximum average delay.

---


## 6. Streamlit Web Dashboard

The project includes an interactive web dashboard in `app.py`:
- **Top Metric Cards**: KPI overview for total deliveries, delivery time, distance, speed, rating, and age.
- **Dynamic Filters**: Sidebar filters by City, Traffic Density, Weather Condition, and Vehicle Condition with instant metric re-calculation.
- **Dedicated Data Cleaning View**: Before vs. After cleaning comparisons, missing value reports, cleaning logic cards, and dataset previews.
- **Competition Questions Tab**: Programmatically computed answers to Q1, Q2, and Q3 with structured data tables.
- **Interactive Visualizations**: Plotly bar charts and scatter plots with trendlines.
- **Executive Insights & AI Summary**: Business recommendations and LLM explanation tab.

---

## 7. Setup & Execution Instructions

> [!NOTE]
> The steps below are provided as execution guidelines. No commands have been automatically run.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment (Optional for AI Summary)
Copy `.env.example` to `.env` and add your API key:
```bash
cp .env.example .env
```

### Step 3: Add Dataset
Place your food delivery CSV file inside the `data/` folder (e.g. `data/food_delivery.csv` or `data/train.csv`).

### Step 4: Run the CLI Pipeline or Streamlit Dashboard

**To run the terminal analysis:**
```bash
python main.py
```

**To run the interactive Streamlit dashboard:**
```bash
streamlit run app.py
```

