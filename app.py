import streamlit as st
import joblib
import numpy as np

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Weather Prediction System",
    page_icon="🌦️",
    layout="wide"
)

# ==========================================================
# Load Models
# ==========================================================

decision_tree = joblib.load("models/decision_tree_model.pkl")
random_forest = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# ==========================================================
# Title
# ==========================================================

st.title("🌦️ Weather Prediction System")
st.markdown("### Predict Weather using Machine Learning")
st.write("This application predicts weather conditions using **Decision Tree** and **Random Forest** algorithms.")

st.divider()

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("⚙️ Prediction Settings")

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    [
        "Decision Tree",
        "Random Forest"
    ]
)

st.sidebar.markdown("---")

precipitation = st.sidebar.number_input(
    "Precipitation",
    min_value=0.0,
    value=0.0,
    step=0.1
)

temp_max = st.sidebar.number_input(
    "Maximum Temperature (°C)",
    value=20.0
)

temp_min = st.sidebar.number_input(
    "Minimum Temperature (°C)",
    value=10.0
)

wind = st.sidebar.number_input(
    "Wind Speed",
    min_value=0.0,
    value=3.0
)

year = st.sidebar.number_input(
    "Year",
    min_value=2012,
    max_value=2035,
    value=2015
)

month = st.sidebar.slider(
    "Month",
    1,
    12,
    7
)

day = st.sidebar.slider(
    "Day",
    1,
    31,
    15
)

weekday = st.sidebar.selectbox(
    "Day of Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

weekday_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

# ==========================================================
# Prediction
# ==========================================================

if st.button("🔍 Predict Weather"):

    data = np.array([[
        precipitation,
        temp_max,
        temp_min,
        wind,
        year,
        month,
        day,
        weekday_map[weekday]
    ]])

    data = scaler.transform(data)

    # Select Model
    if algorithm == "Decision Tree":
        model = decision_tree
    else:
        model = random_forest

    prediction = model.predict(data)
    weather = label_encoder.inverse_transform(prediction)[0]

    # Confidence Score
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(data)) * 100

    # Weather Icons
    icons = {
        "sun": "☀️",
        "rain": "🌧️",
        "snow": "❄️",
        "fog": "🌫️",
        "drizzle": "🌦️"
    }

    icon = icons.get(weather.lower(), "🌍")

    st.success(f"### {icon} Predicted Weather: {weather.upper()}")

    if confidence is not None:
        st.info(f"**Confidence Score:** {confidence:.2f}%")

    st.markdown("---")

    st.subheader("Prediction Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Algorithm:** {algorithm}")
        st.write(f"**Precipitation:** {precipitation}")
        st.write(f"**Maximum Temperature:** {temp_max} °C")
        st.write(f"**Minimum Temperature:** {temp_min} °C")

    with col2:
        st.write(f"**Wind Speed:** {wind}")
        st.write(f"**Date:** {day}-{month}-{year}")
        st.write(f"**Weekday:** {weekday}")
        st.write(f"**Prediction:** {weather.upper()}")

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")
st.caption("Developed by Lakshman Ulli | Weather Prediction using Machine Learning")