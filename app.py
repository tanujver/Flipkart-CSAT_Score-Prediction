import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Flipkart CSAT Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Flipkart Customer Satisfaction Prediction")
st.write("Enter the details below to predict the CSAT score.")

# --------------------------------------------------
# Load Preprocessor and Model
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("CSAT_Preprocessor.joblib")

    model = xgb.XGBClassifier()
    model.load_model("CSAT_XGBoost.json")

    return preprocessor, model


preprocessor, model = load_artifacts()

# --------------------------------------------------
# Extract Categories from OneHotEncoder
# --------------------------------------------------

categorical_pipeline = preprocessor.named_transformers_["cat"]
encoder = categorical_pipeline.named_steps["encoder"]

categorical_columns = [
    "channel_name",
    "category",
    "Sub-category",
    "Tenure Bucket",
    "Agent Shift",
    "Issue_Weekday"
]

category_options = {
    column: encoder.categories_[index].tolist()
    for index, column in enumerate(categorical_columns)
}

# --------------------------------------------------
# Input Form
# --------------------------------------------------

st.subheader("Customer and Issue Details")

with st.form("csat_prediction_form"):

    channel_name = st.selectbox(
        "Channel Name",
        category_options["channel_name"]
    )

    category = st.selectbox(
        "Category",
        category_options["category"]
    )

    sub_category = st.selectbox(
        "Sub-category",
        category_options["Sub-category"]
    )

    tenure_bucket = st.selectbox(
        "Tenure Bucket",
        category_options["Tenure Bucket"]
    )

    agent_shift = st.selectbox(
        "Agent Shift",
        category_options["Agent Shift"]
    )

    issue_weekday = st.selectbox(
        "Issue Weekday",
        category_options["Issue_Weekday"]
    )

    item_price = st.number_input(
        "Item Price",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    response_time = st.number_input(
        "Response Time (Minutes)",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

    issue_hour = st.selectbox(
        "Issue Hour",
        list(range(24))
    )

    issue_day = st.selectbox(
        "Issue Day",
        list(range(1, 32))
    )

    issue_month = st.selectbox(
        "Issue Month",
        list(range(1, 13))
    )

    submit_button = st.form_submit_button(
        "Predict CSAT Score"
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submit_button:

    input_data = pd.DataFrame([{
        "channel_name": channel_name,
        "category": category,
        "Sub-category": sub_category,
        "Item_price": item_price,
        "Tenure Bucket": tenure_bucket,
        "Agent Shift": agent_shift,
        "Response_Time_Minutes": response_time,
        "Issue_Hour": issue_hour,
        "Issue_Day": issue_day,
        "Issue_Month": issue_month,
        "Issue_Weekday": issue_weekday
    }])

    # Apply the saved preprocessing
    transformed_data = preprocessor.transform(input_data)

    # Predict encoded class
    prediction = model.predict(transformed_data)

    # Convert encoded classes 0–4 to CSAT scores 1–5
    predicted_csat = int(prediction[0]) + 1

    st.success(f"Predicted CSAT Score: {predicted_csat} / 5")

    if predicted_csat == 1:
        st.error("Very Dissatisfied 😞")
    elif predicted_csat == 2:
        st.warning("Dissatisfied")
    elif predicted_csat == 3:
        st.info("Neutral")
    elif predicted_csat == 4:
        st.success("Satisfied 😊")
    else:
        st.success("Very Satisfied 😄")