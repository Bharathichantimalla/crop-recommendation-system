
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# Dataset path: same directory as this Python file
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "crop.csv"

st.title("🌱 Crop Recommendation System")
st.write(
    "Enter soil nutrient and weather details "
    "to find a suitable crop."
)

# Load dataset and train model
@st.cache_resource
def train_model(csv_path):
    data = pd.read_csv(csv_path)

    features = [
        "N", "P", "K", "temperature",
        "humidity", "ph", "rainfall"
    ]

    required_columns = features + ["label"]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    X = data[features]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, features, accuracy


# Check whether the dataset exists
if not DATA_PATH.is_file():
    st.error("Dataset file not found.")
    st.write("Expected file location:")
    st.code(str(DATA_PATH))
    st.info(
        "Upload crop.csv to GitHub in the same folder "
        "as app.py. Check the filename carefully."
    )
    st.stop()

try:
    model, features, accuracy = train_model(str(DATA_PATH))

    st.success("Crop recommendation model loaded successfully!")
    st.metric("Model test accuracy", f"{accuracy:.2%}")

    st.subheader("Enter soil and weather details")

    with st.form("crop_recommendation_form"):
        col1, col2 = st.columns(2)

        with col1:
            nitrogen = st.number_input(
                "Nitrogen (N)", min_value=0.0,
                max_value=200.0, value=90.0
            )
            phosphorus = st.number_input(
                "Phosphorus (P)", min_value=0.0,
                max_value=200.0, value=40.0
            )
            potassium = st.number_input(
                "Potassium (K)", min_value=0.0,
                max_value=250.0, value=40.0
            )
            temperature = st.number_input(
                "Temperature (°C)", min_value=-10.0,
                max_value=60.0, value=25.5
            )

        with col2:
            humidity = st.number_input(
                "Humidity (%)", min_value=0.0,
                max_value=100.0, value=80.0
            )
            ph = st.number_input(
                "Soil pH", min_value=0.0,
                max_value=14.0, value=6.5
            )
            rainfall = st.number_input(
                "Rainfall (mm)", min_value=0.0,
                max_value=5000.0, value=200.0
            )

        submitted = st.form_submit_button(
            "🌾 Recommend Crop",
            use_container_width=True
        )

    if submitted:
        input_data = pd.DataFrame(
            [[
                nitrogen,
                phosphorus,
                potassium,
                temperature,
                humidity,
                ph,
                rainfall
            ]],
            columns=features
        )

        prediction = model.predict(input_data)[0]

        st.success(f"Recommended crop: {str(prediction).title()}")
        st.caption(
            "This is a machine-learning prediction based "
            "on your dataset, not a guarantee of crop yield."
        )

except Exception as error:
    st.error("The application could not load the model.")
    st.exception(error)
