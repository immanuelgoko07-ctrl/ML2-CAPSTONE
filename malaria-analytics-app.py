import pandas as pd
import numpy as np
import json

# -----------------------------
# Load and Clean Data
# -----------------------------

def load_and_clean_data():
    df = pd.read_csv("malaria_country_data.csv")

    # Sort data
    df = df.sort_values(["ISO3", "Year"])

    # Fill missing values per country
    numeric_cols = ["Malaria_Cases", "ITN_Coverage", "IRS", "Treatment_Access"]
    df[numeric_cols] = df.groupby("ISO3")[numeric_cols].transform(lambda x: x.ffill().bfill())

    # Remove outliers using IQR (per country)
    def remove_outliers(group):
        Q1 = group["Malaria_Cases"].quantile(0.25)
        Q3 = group["Malaria_Cases"].quantile(0.75)
        IQR = Q3 - Q1
        return group[
            (group["Malaria_Cases"] >= (Q1 - 1.5 * IQR)) &
            (group["Malaria_Cases"] <= (Q3 + 1.5 * IQR))
        ]

    df = df.groupby("ISO3", group_keys=False).apply(remove_outliers)

    return df

# -----------------------------
# Routes
# -----------------------------

def dashboard():
    countries = sorted(df["Country"].unique())
    selected_country = request.form.get("country", countries[0])
    selected_year = request.form.get("year")

    filtered_df = df[df["Country"] == selected_country]

    # Trend Line Chart
    fig_trend = px.line(
        filtered_df,
        x="Year",
        y="Malaria_Cases",
        title=f"Malaria Cases Trend (2007–2017) - {selected_country}",
        markers=True
    )

    graph_trend = json.dumps(fig_trend, cls=plotly.utils.PlotlyJSONEncoder)

    # Prevention vs Cases Scatter
    fig_scatter = px.scatter(
        filtered_df,
        x="ITN_Coverage",
        y="Malaria_Cases",
        size="Treatment_Access",
        color="Year",
        title="ITN Coverage vs Malaria Cases"
    )

    graph_scatter = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)

    # Choropleth Map for Selected Year
    if selected_year:
        year_df = df[df["Year"] == int(selected_year)]
    else:
        year_df = df[df["Year"] == 2017]

    fig_map = px.choropleth(
        year_df,
        locations="ISO3",
        color="Malaria_Cases",
        hover_name="Country",
        color_continuous_scale="Reds",
        title="Malaria Cases Across Africa"
    )

    graph_map = json.dumps(fig_map, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "dashboard.html",
        countries=countries,
        trend=graph_trend,
        scatter=graph_scatter,
        map=graph_map
    )

# -----------------------------
# Run App
# -----------------------------











