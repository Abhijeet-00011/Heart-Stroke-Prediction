import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon=":heart:",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns


model, scaler, expected_columns = load_artifacts()


DEFAULT_VALUES = {
    "age": 40,
    "sex": "M",
    "chest_pain": "ATA",
    "resting_bp": 120,
    "cholesterol": 200,
    "fasting_bs": 0,
    "resting_ecg": "Normal",
    "max_hr": 150,
    "exercise_angina": "N",
    "oldpeak": 1.0,
    "st_slope": "Up",
}


def reset_inputs():
    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def build_model_input(values):
    raw_input = {
        "Age": values["age"],
        "RestingBP": values["resting_bp"],
        "Cholesterol": values["cholesterol"],
        "FastingBS": values["fasting_bs"],
        "MaxHR": values["max_hr"],
        "Oldpeak": values["oldpeak"],
        "Sex_" + values["sex"]: 1,
        "ChestPainType_" + values["chest_pain"]: 1,
        "RestingECG_" + values["resting_ecg"]: 1,
        "ExerciseAngina_" + values["exercise_angina"]: 1,
        "ST_Slope_" + values["st_slope"]: 1,
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    return input_df[expected_columns]


def get_risk_probability(scaled_input):
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(scaled_input)[0]
    classes = list(getattr(model, "classes_", []))

    if 1 in classes:
        risk_index = classes.index(1)
    else:
        risk_index = len(probabilities) - 1

    return float(probabilities[risk_index])


def collect_warnings(values):
    warnings = []

    if values["resting_bp"] >= 180:
        warnings.append("Resting blood pressure is very high.")
    elif values["resting_bp"] >= 140:
        warnings.append("Resting blood pressure is above the normal range.")

    if values["cholesterol"] >= 240:
        warnings.append("Cholesterol is high.")

    if values["oldpeak"] >= 2.5:
        warnings.append("Oldpeak value is elevated.")

    if values["max_hr"] < 90:
        warnings.append("Maximum heart rate is unusually low.")

    return warnings


for key, value in DEFAULT_VALUES.items():
    st.session_state.setdefault(key, value)

st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1120px;
        }
        .app-title {
            font-size: 2.2rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
            color: #1f2937;
        }
        .app-subtitle {
            color: #4b5563;
            font-size: 1.02rem;
            margin-bottom: 1.4rem;
        }
        .result-box {
            border-radius: 8px;
            padding: 1.1rem 1.25rem;
            border: 1px solid;
            margin-top: 1rem;
        }
        .result-high {
            background: #fff1f2;
            border-color: #fb7185;
            color: #881337;
        }
        .result-low {
            background: #ecfdf5;
            border-color: #34d399;
            color: #064e3b;
        }
        .result-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .result-meta {
            font-size: 0.98rem;
            line-height: 1.55;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("About")
st.sidebar.info(
    "This app estimates heart disease risk using a trained KNN model. "
    "It is for learning and screening support only, not a medical diagnosis."
)

with st.sidebar.expander("Model details", expanded=True):
    st.write("Algorithm: K-Nearest Neighbors")
    st.write("Inputs: clinical and ECG-related measurements")
    st.write("Output: low or high predicted risk")

with st.sidebar.expander("Abbreviations"):
    st.write("ATA: Atypical Angina")
    st.write("NAP: Non-Anginal Pain")
    st.write("TA: Typical Angina")
    st.write("ASY: Asymptomatic")
    st.write("LVH: Left Ventricular Hypertrophy")
    st.write("ST: ST-T wave abnormality")

if st.sidebar.button("Reset inputs", use_container_width=True):
    reset_inputs()
    rerun_app()

if st.sidebar.button("Clear prediction history", use_container_width=True):
    st.session_state["prediction_history"] = []
    rerun_app()

st.markdown('<div class="app-title">Heart Disease Risk Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Enter the patient details below and generate a quick risk estimate.</div>',
    unsafe_allow_html=True,
)

sex_labels = {
    "M": "Male",
    "F": "Female",
}

chest_pain_labels = {
    "ATA": "ATA - Atypical Angina",
    "NAP": "NAP - Non-Anginal Pain",
    "TA": "TA - Typical Angina",
    "ASY": "ASY - Asymptomatic",
}

fasting_bs_labels = {
    0: "No - 120 mg/dL or below",
    1: "Yes - above 120 mg/dL",
}

resting_ecg_labels = {
    "Normal": "Normal",
    "ST": "ST - ST-T wave abnormality",
    "LVH": "LVH - Left Ventricular Hypertrophy",
}

exercise_angina_labels = {
    "N": "No",
    "Y": "Yes",
}

st_slope_labels = {
    "Up": "Up - Upsloping",
    "Flat": "Flat",
    "Down": "Down - Downsloping",
}

with st.form("prediction_form"):
    st.subheader("Patient Details")
    personal_col, pain_col = st.columns(2)

    with personal_col:
        age = st.slider(
            "Age",
            18,
            100,
            key="age",
            help="Patient age in years.",
        )
        sex = st.selectbox(
            "Sex",
            ["M", "F"],
            key="sex",
            format_func=lambda option: sex_labels[option],
        )

    with pain_col:
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "TA", "ASY"],
            key="chest_pain",
            format_func=lambda option: chest_pain_labels[option],
            help="Type of chest pain reported by the patient.",
        )
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar",
            [0, 1],
            key="fasting_bs",
            format_func=lambda option: fasting_bs_labels[option],
            help="Whether fasting blood sugar is above 120 mg/dL.",
        )

    st.subheader("Clinical Measurements")
    measure_col, heart_col = st.columns(2)

    with measure_col:
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=220,
            step=1,
            key="resting_bp",
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=100,
            max_value=700,
            step=1,
            key="cholesterol",
        )

    with heart_col:
        max_hr = st.slider(
            "Maximum Heart Rate",
            60,
            220,
            key="max_hr",
        )
        oldpeak = st.slider(
            "Oldpeak (ST Depression)",
            0.0,
            6.0,
            step=0.1,
            key="oldpeak",
            help="ST depression induced by exercise relative to rest.",
        )

    st.subheader("ECG and Exercise Details")
    ecg_col, exercise_col = st.columns(2)

    with ecg_col:
        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"],
            key="resting_ecg",
            format_func=lambda option: resting_ecg_labels[option],
        )

    with exercise_col:
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["N", "Y"],
            key="exercise_angina",
            format_func=lambda option: exercise_angina_labels[option],
        )
        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"],
            key="st_slope",
            format_func=lambda option: st_slope_labels[option],
        )

    submitted = st.form_submit_button("Predict Risk", use_container_width=True)

if submitted:
    values = {
        "age": age,
        "sex": sex,
        "chest_pain": chest_pain,
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "fasting_bs": fasting_bs,
        "resting_ecg": resting_ecg,
        "max_hr": max_hr,
        "exercise_angina": exercise_angina,
        "oldpeak": oldpeak,
        "st_slope": st_slope,
    }

    input_df = build_model_input(values)
    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])
    risk_probability = get_risk_probability(scaled_input)
    probability_text = (
        f"{risk_probability:.0%}" if risk_probability is not None else "Not available"
    )

    st.subheader("Prediction Result")

    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-box result-high">
                <div class="result-title">High Risk of Heart Disease</div>
                <div class="result-meta">
                    Estimated risk probability: <strong>{probability_text}</strong><br>
                    Please consult a qualified medical professional for proper evaluation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-box result-low">
                <div class="result-title">Low Risk of Heart Disease</div>
                <div class="result-meta">
                    Estimated risk probability: <strong>{probability_text}</strong><br>
                    Keep following healthy habits and consult a professional if symptoms exist.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    warnings = collect_warnings(values)
    if warnings:
        st.warning("Please review: " + " ".join(warnings))

    history_row = {
        "Age": age,
        "Sex": sex_labels[sex],
        "Chest Pain": chest_pain_labels[chest_pain],
        "Blood Pressure": resting_bp,
        "Cholesterol": cholesterol,
        "Max HR": max_hr,
        "Prediction": "High Risk" if prediction == 1 else "Low Risk",
        "Risk Probability": probability_text,
    }

    st.session_state.setdefault("prediction_history", [])
    st.session_state["prediction_history"].insert(0, history_row)
    st.session_state["prediction_history"] = st.session_state["prediction_history"][:10]

if st.session_state.get("prediction_history"):
    st.subheader("Recent Predictions")
    st.dataframe(
        pd.DataFrame(st.session_state["prediction_history"]),
        use_container_width=True,
        hide_index=True,
    )
