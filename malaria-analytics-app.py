import pandas as pd
import numpy as np
import plotly
import json
import os

app = Flask(__name__)

DATA_PATH = "malaria_country_data.csv"

# -------------------------------------------------
# Load & Clean Data
# -------------------------------------------------

def load_and_clean_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset not found. Ensure malaria_country_data.csv exists.")

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "ISO3", "Country", "Year",
        "Malaria_Cases", "ITN_Coverage",
        "IRS", "Treatment_Access"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.sort_values(["ISO3", "Year"]).reset_index(drop=True)

    # Fill missing numeric values per country
    numeric_cols = ["Malaria_Cases", "ITN_Coverage", "IRS", "Treatment_Access"]
    df[numeric_cols] = df.groupby("ISO3")[numeric_cols].transform(
        lambda x: x.ffill().bfill()
    )

    # Remove outliers using IQR (per country)
    cleaned_groups = []

    for country, group in df.groupby("ISO3"):
        Q1 = group["Malaria_Cases"].quantile(0.25)
        Q3 = group["Malaria_Cases"].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        filtered = group[
            (group["Malaria_Cases"] >= lower) &
            (group["Malaria_Cases"] <= upper)
        ]

        cleaned_groups.append(filtered)

    df_clean = pd.concat(cleaned_groups).reset_index(drop=True)

    return df_clean


df = load_and_clean_data()


# -------------------------------------------------
# Dashboard Route
# -------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def dashboard():

    countries = sorted(df["Country"].unique())
    years = sorted(df["Year"].unique())

    selected_country = countries[0]
    selected_year = years[-1]

    if request.method == "POST":
        selected_country = request.form.get("country", selected_country)
        selected_year = int(request.form.get("year", selected_year))

    # Filter by selected country
    country_df = df[df["Country"] == selected_country]

    # ---------------- Trend Chart ----------------
    fig_trend = px.line(
        country_df,
        x="Year",
        y="Malaria_Cases",
        markers=True,
        title=f"Malaria Cases Trend (2007–2017) - {selected_country}"
    )

    trend_graph = json.dumps(fig_trend, cls=plotly.utils.PlotlyJSONEncoder)

    # ---------------- Scatter Plot ----------------
    fig_scatter = px.scatter(
        country_df,
        x="ITN_Coverage",
        y="Malaria_Cases",
        size="Treatment_Access",
        color="Year",
        title="ITN Coverage vs Malaria Cases",
        hover_data=["Year"]
    )

    scatter_graph = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)

    # ---------------- Map ----------------
    year_df = df[df["Year"] == selected_year]

    fig_map = px.choropleth(
        year_df,
        locations="ISO3",
        color="Malaria_Cases",
        hover_name="Country",
        color_continuous_scale="Reds",
        projection="natural earth",
        title=f"Malaria Cases Across Africa ({selected_year})"
    )

    map_graph = json.dumps(fig_map, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "dashboard.html",
        countries=countries,
        years=years,
        selected_country=selected_country,
        selected_year=selected_year,
        trend_graph=trend_graph,
        scatter_graph=scatter_graph,
        map_graph=map_graph
    )


# -------------------------------------------------
# Run App
# -------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)


