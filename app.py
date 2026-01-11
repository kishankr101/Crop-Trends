
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(layout="wide")
st.title('🌾 AI-Based Crop Production Trend Prediction')

@st.cache_data
def load_data(file_path):
    df = pd.read_csv("/content/crop-data.csv")
    # Clean column names to remove BOM characters and quotes
    df.columns = df.columns.str.replace('ï»¿', '', regex=False).str.replace('"', '', regex=False).str.strip()
    return df

@st.cache_data
def preprocess_data(df, selected_crop, selected_state):
    # Filter to MVP scope (example: Rice in Andhra Pradesh) and 'Total' season
    df_filtered = df[(df['Crop'] == selected_crop) & (df['State'] == selected_state)].copy()

    if df_filtered.empty:
        return pd.DataFrame()

    # Identify production columns for melting
    production_cols = [col for col in df_filtered.columns if 'Production-' in col]

    # Melt the DataFrame to transform year-specific columns into 'Year_Col' and 'Production' columns
    df_melted_cleaned = df_filtered.melt(id_vars=['Crop', 'State', 'Season'],
                                         value_vars=production_cols,
                                         var_name='Year_Col',
                                         value_name='Production')

    # Extract the year from the 'Year_Col' (e.g., 'Production-2021-22' -> 2021)
    df_melted_cleaned['Year'] = df_melted_cleaned['Year_Col'].str.extract(r'(\d{4})').astype(int)

    # Remove rows with missing production values
    df_melted_cleaned.dropna(subset=['Production'], inplace=True)

    # Sort by year for time-series consistency
    df_melted_cleaned = df_melted_cleaned.sort_values(by=['Season', 'Year'])

    return df_melted_cleaned

@st.cache_data
def train_models(df_long):
    results = {}
    if df_long.empty:
        return results

    for season in df_long["Season"].unique():
        df_s = df_long[df_long["Season"] == season]

        if df_s.shape[0] < 2: # Need at least 2 data points for linear regression
            st.warning(f"Not enough data to train model for {season} season. Skipping.")
            continue

        X = df_s[["Year"]]
        y = df_s["Production"]

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        results[season] = {
            "model": model,
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
            "last_production": y.iloc[-1]
        }
    return results

@st.cache_data
def predict_future(_results, future_years_df):
    all_predictions = []
    if not _results:
        return pd.DataFrame()

    for season, info in _results.items():
        preds = info["model"].predict(future_years_df)
        for i, year in enumerate(future_years_df['Year']):
            all_predictions.append({
                'Season': season,
                'Year': year,
                'Predicted_Production': preds[i]
            })
    return pd.DataFrame(all_predictions)

@st.cache_data
def generate_insights(_results, df_predictions_forecast, future_years_df):
    forecast = []
    if not _results:
        return pd.DataFrame()

    for season, info in _results.items():
        # Filter predictions for the current season
        season_future_predictions_df = df_predictions_forecast[df_predictions_forecast['Season'] == season]
        if season_future_predictions_df.empty:
            continue

        avg_future = season_future_predictions_df['Predicted_Production'].mean()

        # Determine the status based on a 5% threshold compared to the last historical production
        if info.get("last_production") is not None and info["last_production"] > 0:
            if avg_future > info["last_production"] * 1.05:
                status = "Likely Surplus"
            elif avg_future < info["last_production"] * 0.95:
                status = "Likely Shortage"
            else:
                status = "Balanced"
        else:
            status = "N/A (No historical data or zero last production)"

        forecast.append({
            "Season": season,
            "Avg_Future_Production": round(avg_future, 2),
            "Status": status
        })
    return pd.DataFrame(forecast)

def plot_production_trends(df_long_historical, df_predictions_forecast, selected_state, selected_crop):
    if df_long_historical.empty and df_predictions_forecast.empty:
        return None

    df_long_plot = df_long_historical.copy()
    df_long_plot['DataType'] = 'Historical'

    df_predictions_plot = df_predictions_forecast.copy()
    df_predictions_plot.rename(columns={'Predicted_Production': 'Production'}, inplace=True)
    df_predictions_plot['DataType'] = 'Predicted'

    combined_df = pd.concat([df_long_plot, df_predictions_plot], ignore_index=True)
    combined_df = combined_df.sort_values(by=['Season', 'Year']).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=combined_df, x='Year', y='Production', hue='Season', style='DataType', marker='o', ax=ax)
    ax.set_title(f'Historical and Predicted {selected_crop} Production Trends by Season - {selected_state}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Production (million tonnes)')
    ax.grid(True)
    ax.legend(title='Season/Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticks(combined_df['Year'].unique())
    plt.tight_layout()
    return fig

# --- Main Streamlit App Logic ---

# Corrected file_path
file_path = '/content/crop-data.csv'

try:
    original_df = load_data(file_path)
except FileNotFoundError:
    st.error(f"Dataset file not found. Please ensure '{file_path}' is in the correct directory.")
    st.stop()

# Sidebar for user selection
st.sidebar.header('Configuration')

available_crops = original_df['Crop'].unique()
selected_crop = st.sidebar.selectbox('Select Crop', available_crops, index=list(available_crops).index('Rice'))

# Filter states based on the selected crop
available_states = original_df[original_df['Crop'] == selected_crop]['State'].unique()
selected_state = st.sidebar.selectbox('Select State', available_states, index=list(available_states).index('Andhra Pradesh'))

# Preprocess data
df_long = preprocess_data(original_df, selected_crop, selected_state)

if df_long.empty:
    st.warning(f"No historical data found for {selected_crop} in {selected_state} with valid production values.")
else:
    st.subheader('1. Historical Production Data (Andhra Pradesh, Rice)')
    st.dataframe(df_long)

    st.subheader('2. Historical Production Trends')
    fig_hist = plot_production_trends(df_long, pd.DataFrame(), selected_state, selected_crop) # No predictions yet for this plot
    if fig_hist:
        st.pyplot(fig_hist)

    # Feature Engineering & Model Training
    st.subheader('3. Model Training & Evaluation')
    results = train_models(df_long)

    if results:
        st.write("Linear Regression models trained for each season.")
        st.dataframe(pd.DataFrame([
            {"Season": s, "MAE": r["mae"], "R2_Score": r["r2"], "Last_Historical_Production": r["last_production"]}
            for s, r in results.items()
        ]))

        # Prediction & Insights
        st.subheader('4. Future Production Forecast')
        future_years_df = pd.DataFrame({'Year': [2026, 2027, 2028, 2029, 2030]})
        df_predictions = predict_future(results, future_years_df)

        if not df_predictions.empty:
            st.write(f"Predicted {selected_crop} production for {selected_state} (2026-2030):")
            st.dataframe(df_predictions)

            # Generate insights
            forecast_df = generate_insights(results, df_predictions, future_years_df)
            if not forecast_df.empty:
                st.subheader('5. Production Outlook (Surplus/Shortage)')
                st.dataframe(forecast_df)

                st.subheader('6. Actionable Recommendations')
                for _, row in forecast_df.iterrows():
                    st.markdown(f"- **{row['Season']} season**: Expected to show **{row['Status']}** with average future production of **{row['Avg_Future_Production']}** million tonnes.")
                    if row['Status'] == 'Likely Surplus':
                        st.info(f"  Consider exploring new markets, storage solutions, or diversification for {row['Season']} season.")
                    elif row['Status'] == 'Likely Shortage':
                        st.warning(f"  Strategize to increase yield, explore alternative sources, or manage demand for {row['Season']} season.")
                    else:
                        st.success(f"  Maintain current practices and monitor conditions closely for {row['Season']} season.")

                # Visualization of combined trends
                st.subheader('7. Historical & Predicted Trends Visualization')
                fig_combined = plot_production_trends(df_long, df_predictions, selected_state, selected_crop)
                if fig_combined:
                    st.pyplot(fig_combined)

                # Save combined data to CSV (optional, for local use or download)
                combined_df_final = pd.concat([
                    df_long.assign(DataType='Historical'),
                    df_predictions.rename(columns={'Predicted_Production': 'Production'}).assign(DataType='Predicted')
                ], ignore_index=True)
                combined_df_final = combined_df_final.sort_values(by=['Season', 'Year']).reset_index(drop=True)
                st.download_button(
                    label="Download Combined Data CSV",
                    data=combined_df_final.to_csv(index=False).encode('utf-8'),
                    file_name=f'combined_production_forecast_{selected_crop.replace(" ", "_")}_{selected_state.replace(" ", "_")}.csv',
                    mime='text/csv',
                )
            else:
                st.warning("Could not generate forecast insights.")
        else:
            st.warning("Could not generate future predictions.")
    else:
        st.warning("No models were successfully trained. Please check the data.")
