import streamlit as st
import pickle
import pandas as pd
import sklearn


# Cache the model so it only loads once, saving memory and time
@st.cache_resource
def load_model():
    with open("my_model.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()

# Have app fit screen and add a browser tab title/icon
st.set_page_config(page_title="Loan Approval Form", page_icon="🏦", layout="wide")

# Native Streamlit headers generally look cleaner and adapt to dark/light mode automatically
st.title("🏦 Loan Approval Application")
st.markdown("Please fill out the applicant's details below to predict loan approval status")
st.divider()

# Dictionary mappings to show clean text to the user but send exact strings to the model
employment_status_map = {
    "Full Time": "full_time",
    "Part Time": "part_time",
    "Unemployed": "unemployed"
}

employment_sector_map = {
    "Unknown": "unknown",
    "Communication Services": "communication_services",
    "Consumer Discretionary": "consumer_discretionary",
    "Consumer Staples": "consumer_staples",
    "Energy": "energy",
    "Financials": "financials",
    "Health Care": "health_care",
    "Industrials": "industrials",
    "Information Technology": "information_technology",
    "Materials": "materials",
    "Real Estate": "real_estate",
    "Utilities": "utilities"
}

reason_map = {
    "Cover an Unexpected Cost": "cover_an_unexpected_cost",
    "Credit Card Refinancing": "credit_card_refinancing",
    "Debt Consolidation": "debt_conslidation",  # Kept the original spelling to ensure model compatibility
    "Home Improvement": "home_improvement",
    "Major Purchase": "major_purchase",
    "Other": "other"
}

# st.form prevents the page from refreshing on every single slider movement
with st.form("loan_application_form"):
    # Arrange inputs into 3 neat columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Loan Details")
        requested_loan_amount = st.number_input("Requested Loan Amount ($)", min_value=5000, max_value=125000,
                                                step=1000)
        Reason_UI = st.radio("Reason for Loan", list(reason_map.keys()))

    with col2:
        st.subheader("Financial Profile")
        Monthly_Gross_Income = st.slider("Monthly Gross Income ($)", min_value=-2559, max_value=14005, value=4000)
        Monthly_Housing_Payment = st.slider("Monthly Housing Payment ($)", min_value=300, max_value=3300, value=1000)

    with col3:
        st.subheader("Credit & Employment")
        FICO_score = st.slider("FICO Score", min_value=373, max_value=850, value=700)
        Employment_Status_UI = st.selectbox("Employment Status", list(employment_status_map.keys()))
        Employment_Sector_UI = st.selectbox("Employment Sector", list(employment_sector_map.keys()))
        Ever_Bankrupt = st.checkbox("Has the applicant ever filed for bankruptcy?")

    # Submit button for the form
    submitted = st.form_submit_button("Predict Loan Approval", type="primary")

# Only run calculations and predictions if the user clicks the button
if submitted:
    # Prevent ZeroDivisionError if income is exactly 0
    safe_income = Monthly_Gross_Income if Monthly_Gross_Income != 0 else 0.01

    DTI = Monthly_Housing_Payment / safe_income
    Loan_to_Income_Ratio = requested_loan_amount / safe_income
    Monthly_Disposable_Income = Monthly_Gross_Income - Monthly_Housing_Payment
    Loan_to_Fico_Ratio = requested_loan_amount / FICO_score

    # Create the input DataFrame, mapping the UI selections back to the model's required strings
    input_data = pd.DataFrame({
        "Requested_Loan_Amount": [requested_loan_amount],
        "FICO_score": [FICO_score],
        "Monthly_Gross_Income": [Monthly_Gross_Income],
        "Monthly_Housing_Payment": [Monthly_Housing_Payment],
        "DTI": [DTI],
        "Loan_to_Income_Ratio": [Loan_to_Income_Ratio],
        "Monthly_Disposable_Income": [Monthly_Disposable_Income],
        "Loan_to_Fico_Ratio": [Loan_to_Fico_Ratio],
        "Ever_Bankrupt_or_Foreclose": [int(Ever_Bankrupt)],
        "Employment_Status": [employment_status_map[Employment_Status_UI]],
        "Employment_Sector": [employment_sector_map[Employment_Sector_UI]],
        "Reason": [reason_map[Reason_UI]]
    })

    # Add a visual spinner while the model runs
    with st.spinner("Analyzing applicant profile..."):
        try:
            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.success("✅ Loan Approved!")
            else:
                st.error("❌ Loan Denied")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")