# 🌾 AI‑Based Crop Production Trend Predictor

Agriculture lacks predictive decision‑support systems that help farmers and planners
anticipate crop surplus and shortages before they occur.

This project uses AI and historical crop production data to predict future trends
and provide early warnings for potential surplus or shortage situations.

---

## 🔍 Problem Statement
Farmers and agricultural planners currently rely on historical data and reactive
measures. There is no simple system that provides forward‑looking insights to support
crop planning decisions.

---

## 💡 Solution Overview
Our system analyzes past crop production data and uses machine learning to:
- Learn production trends over time
- Predict future crop production
- Identify surplus or shortage risks
- Provide actionable recommendations

The goal is to enable **early decision‑making instead of late reaction**.

---

## ⚙️ How It Works
1. Load historical crop production data
2. Clean and preprocess year‑wise values
3. Train a trend‑based ML model (Linear Regression)
4. Predict future production
5. Classify the trend and risk level
6. Display insights via an interactive Streamlit app

---

## 🖥️ Application Features
- Select State and Crop
- Predict future production trends
- Visualize historical and forecasted data
- Get clear surplus / shortage indicators
- Receive simple planning recommendations

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit‑Learn
- Streamlit
- Google Colab (development)

---

## 📊 Dataset
The dataset contains:
- Crop name
- State
- Season
- Area, Production, and Yield across multiple years

Source: Government agricultural statistics (processed for analysis)

---

## 🚫 Out of Scope (MVP)
- Weather forecasting
- Satellite imagery
- Real‑time market prices
- Farmer‑level personalization

These can be added in future versions.

---

## 🔮 Future Scope
- Weather and climate integration
- Market price prediction
- Multi‑crop analysis
- Mobile‑friendly interface
- Policy‑level planning dashboards

---

## ⚠️ Disclaimer
This system provides trend‑based predictions for decision support.
Results are indicative and should be used along with expert agricultural guidance.

---

## 🚀 Deployment
The application is deployed using **Streamlit Cloud**.

---

## 👥 Team
- Kishan Kumar
- V. Mohan
- J. Leo
