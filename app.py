import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cardiac Risk Prediction",
    page_icon="❤️",
    layout="centered"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("models/cardiac_model.pkl")


# ============================================================
# TITLE
# ============================================================

st.title("❤️ Cardiac Risk Prediction")

st.write(
    "Enter the patient's clinical information below "
    "to generate a machine-learning prediction."
)

st.caption(
    "This application is intended for educational and "
    "demonstration purposes."
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.header("Patient Information")

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN — PATIENT DETAILS
# ============================================================

with col1:

    st.subheader("Patient Details")

    # Age
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=50
    )

    # Sex
    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x:
        "Female" if x == 0 else "Male"
    )

    # Chest Pain Type
    cp_label = st.selectbox(
        "Chest Pain Type",
        options=[
            "Typical angina",
            "Atypical angina",
            "Non-anginal pain",
            "Asymptomatic"
        ]
    )

    # Convert UI label to model value
    cp = {
        "Typical angina": 0,
        "Atypical angina": 1,
        "Non-anginal pain": 2,
        "Asymptomatic": 3
    }[cp_label]

    # Fasting Blood Sugar
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )

    # Resting ECG
    restecg_label = st.selectbox(
        "Resting ECG",
        options=[
            "Normal",
            "ST-T wave abnormality",
            "Left ventricular hypertrophy"
        ]
    )

    # Convert UI label to model value
    restecg = {
        "Normal": 0,
        "ST-T wave abnormality": 1,
        "Left ventricular hypertrophy": 2
    }[restecg_label]

    # Exercise Induced Angina
    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )


# ============================================================
# RIGHT COLUMN — CLINICAL MEASUREMENTS
# ============================================================

with col2:

    st.subheader("Clinical Measurements")

    # Resting Blood Pressure
    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    # Cholesterol
    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=600,
        value=200
    )

    # Maximum Heart Rate
    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    # ST Depression
    oldpeak = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    # Slope
    slope_label = st.selectbox(
        "ST Segment Slope",
        options=[
            "Upsloping",
            "Flat",
            "Downsloping"
        ]
    )

    # Convert UI label to model value
    slope = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }[slope_label]

    # Number of Major Vessels
    ca = st.selectbox(
        "Number of Major Vessels",
        options=[0, 1, 2, 3, 4]
    )

    # Thalassemia
    thal_label = st.selectbox(
        "Thalassemia",
        options=[
            "Normal",
            "Fixed defect",
            "Reversible defect",
            "Unknown"
        ]
    )

    # Convert UI label to model value
    thal = {
        "Normal": 0,
        "Fixed defect": 1,
        "Reversible defect": 2,
        "Unknown": 3
    }[thal_label]


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Cardiac Risk",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create patient DataFrame
    # --------------------------------------------------------

    patient = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })


    # --------------------------------------------------------
    # Model Prediction
    # --------------------------------------------------------

    prediction = model.predict(patient)[0]

    probability = model.predict_proba(patient)[0][1]


    # --------------------------------------------------------
    # Prediction Result
    # --------------------------------------------------------

    st.divider()

    st.header("Prediction Result")

    st.metric(
        label="Estimated Positive-Class Probability",
        value=f"{probability * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Prediction Message
    # --------------------------------------------------------

    if prediction == 1:

        st.warning(
            "⚠️ Higher Risk Prediction"
        )

    else:

        st.success(
            "✅ Lower Risk Prediction"
        )


    # --------------------------------------------------------
    # Probability Visualization
    # --------------------------------------------------------

    st.progress(
        float(probability)
    )


    # --------------------------------------------------------
    # Model Information
    # --------------------------------------------------------

    with st.expander("ℹ️ About This Model"):

        st.write("**Algorithm:** Random Forest Classifier")

        st.write(
            "**Preprocessing:** ColumnTransformer"
        )

        st.write(
            "**Numerical preprocessing:** "
            "Median imputation + StandardScaler"
        )

        st.write(
            "**Categorical preprocessing:** "
            "Most-frequent imputation + OneHotEncoder"
        )

        st.write(
            "**Hyperparameter tuning:** GridSearchCV"
        )

        st.write(
            "**Validation:** Stratified K-Fold Cross-Validation"
        )


    # --------------------------------------------------------
    # Disclaimer
    # --------------------------------------------------------

    st.info(
        "This is a machine-learning prediction based on "
        "the trained dataset and model. It is not a medical "
        "diagnosis and should not be used as a substitute "
        "for professional medical advice."
    )