import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Credit Score Dashboard",
    layout="wide",
    page_icon="💳"
)

st.title("💳 Credit Score Prediction Dashboard")
st.divider()

# ==========================================
# 2. Caching Resources (Model & Scaler)
# ==========================================
@st.cache_resource
def load_credit_model():
    if os.path.exists("credit_model.pkl"):
        loaded = joblib.load("credit_model.pkl")
        return loaded[0] if isinstance(loaded, (list, tuple)) else loaded
    return None

@st.cache_resource
def load_figure(file_name):
    if os.path.exists(file_name):
        return joblib.load(file_name)
    return None

@st.cache_data
def load_scaling_parameters():
    if os.path.exists("scaling_parameters.csv"):
        return pd.read_csv("scaling_parameters.csv")
    return pd.DataFrame()

model = load_credit_model()
scaling_params = load_scaling_parameters()

# ==========================================
# 3. Top Tabs Navigation
# ==========================================
tab_eda, tab_metrics, tab_predict = st.tabs([
    "📊 Exploratory Data Analysis (EDA)",
    "📈 Model Evaluation & Metrics",
    "🔮 Model Testing & Prediction"
])

# ==========================================
# TAB 1: EDA Charts (With Spinner & Dynamic Height)
# ==========================================
with tab_eda:
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.write("Select any visualization below to inspect its distribution interactively.")

    # Explicit dictionary of charts
    chart_options = {
        "Chart 01: Credit Score Count (Target Distribution)": "credit1.pkl",
        "Chart 02: Numerical Distributions": "credit2.pkl",
        "Chart 03: Age Boxplot": "credit3.pkl",
        "Chart 04: Delay From Due Date Boxplot": "credit4.pkl",
        "Chart 05: Age by Credit Score": "credit5.pkl",
        "Chart 06: Credit History Age by Credit Score": "credit6.pkl",
        "Chart 07: Outstanding Debt by Credit Score": "credit7.pkl",
        "Chart 08: Number of Credit Inquiries by Credit Score": "credit8.pkl",
        "Chart 09: Number of Delayed Payments by Credit Score": "credit9.pkl",
        "Chart 10: Delay From Due Date by Credit Score": "credit10.pkl",
        "Chart 11: Debt to Income by Credit Score": "credit11.pkl",
        "Chart 12: Credit Mix Distribution": "credit12.pkl",
        "Chart 13: Number of Credit Mix by Credit Score": "credit13.pkl",
        "Chart 14: Occupation Distribution": "credit14.pkl",
        "Chart 15: Number of Occupations by Credit Score": "credit15.pkl",
        "Chart 16: Month Distribution": "credit16.pkl",
        "Chart 17: Number of Months by Credit Score": "credit17.pkl"
    }

    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        selected_chart_name = st.selectbox("📌 Select Chart to Display:", list(chart_options.keys()))
    with col_ctrl2:
        render_all = st.checkbox("Show All Charts at Once", value=False)

    if render_all:
        with st.spinner("⏳ Loading all visualizations... Please wait a moment."):
            for name, file_path in chart_options.items():
                fig = load_figure(file_path)
                if fig is not None:
                    st.subheader(name)
                    # Double height for Age Boxplot or credit2.pkl
                    is_extended = (file_path in ["credit2.pkl", "credit3.pkl"] or "age" in name.lower())
                    custom_height = 1200 if is_extended else 550
                    fig.update_layout(height=custom_height, autosize=True)
                    st.plotly_chart(fig, key=f"all_{file_path}", use_container_width=True)
    else:
        file_to_render = chart_options[selected_chart_name]
        with st.spinner(f"⏳ Loading {selected_chart_name}... Please wait."):
            fig = load_figure(file_to_render)
            if fig is not None:
                # Double height for Age Boxplot or credit2.pkl
                is_extended = (file_to_render in ["credit2.pkl", "credit3.pkl"] or "age" in selected_chart_name.lower())
                custom_height = 1300 if is_extended else 650
                fig.update_layout(height=custom_height, autosize=True)
                st.plotly_chart(fig, key=f"single_{file_to_render}", use_container_width=True)
            else:
                st.warning(f"File `{file_to_render}` not found.")

# ==========================================
# TAB 2: Model Evaluation & Metrics
# ==========================================
with tab_metrics:
    st.header("📈 Model Evaluation & Confusion Matrices")
    st.write("Performance evaluation metrics for initial and feature-selected models.")

    with st.spinner("⏳ Loading evaluation matrices and importance charts..."):
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.subheader("Testing Confusion Matrix")
            fig18 = load_figure("credit18.pkl")
            if fig18:
                fig18.update_layout(height=450, autosize=True)
                st.plotly_chart(fig18, key="tab2_credit18", use_container_width=True)
                st.info("Testing Accuracy: **75%**")

        with col_m2:
            st.subheader("Training Confusion Matrix")
            fig19 = load_figure("credit19.pkl")
            if fig19:
                fig19.update_layout(height=450, autosize=True)
                st.plotly_chart(fig19, key="tab2_credit19", use_container_width=True)
                st.info("Training Accuracy: **80%**")

        st.markdown("---")
        st.subheader("Feature Importance")
        feat_fig_file = "feature_importance.pkl" if os.path.exists("feature_importance.pkl") else "credit20.pkl"
        fig_feat = load_figure(feat_fig_file)
        if fig_feat:
            fig_feat.update_layout(height=550, autosize=True)
            st.plotly_chart(fig_feat, key="tab2_feat", use_container_width=True)

        st.markdown("---")
        col_m3, col_m4 = st.columns(2)

        with col_m3:
            st.subheader("Retrained Testing Matrix (Selected Features)")
            fig21 = load_figure("credit21.pkl")
            if fig21:
                fig21.update_layout(height=450, autosize=True)
                st.plotly_chart(fig21, key="tab2_credit21", use_container_width=True)
                st.info("Testing Accuracy: **76%**")

        with col_m4:
            st.subheader("Retrained Training Matrix (Selected Features)")
            fig22 = load_figure("credit22.pkl")
            if fig22:
                fig22.update_layout(height=450, autosize=True)
                st.plotly_chart(fig22, key="tab2_credit22", use_container_width=True)
                st.info("Training Accuracy: **80%**")

# ==========================================
# TAB 3: Model Testing & Prediction
# ==========================================
with tab_predict:
    st.header("🔮 Customer Profile Credit Prediction")
    st.write("Input customer financial parameters to test model output.")

    if model is None:
        st.error("⚠️ Model file `credit_model.pkl` could not be loaded.")
    else:
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            Credit_History_years = st.number_input("Credit History (Years)", min_value=0.0, max_value=60.0, value=5.0, step=1.0)
            Credit_History_months = st.number_input("Credit History (Months)", min_value=0.0, max_value=11.0, value=6.0, step=1.0)
            Outstanding_Debt = st.number_input("Outstanding Debt ($)", min_value=0.0, value=1200.0, step=100.0)
            Num_Credit_Inquiries = st.number_input("Number of Credit Inquiries", min_value=0.0, max_value=50.0, value=3.0, step=1.0)
            Annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)

        with col_in2:
            Delay_from_due_date = st.number_input("Delay from Due Date (Days)", min_value=0.0, max_value=120.0, value=5.0, step=1.0)
            Interest_Rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.5)
            Num_Bank_Accounts = st.number_input("Number of Bank Accounts", min_value=0.0, max_value=30.0, value=3.0, step=1.0)
            Payment_of_Min_Amount = st.selectbox("Payment of Minimum Amount", ["Yes", "No", "Not_mensioned"])
            Credit_Mix = st.selectbox("Credit Mix", ["Good", "Standard", "Bad", "undetermined_credit_mix"])

        Credit_History_Age = (Credit_History_years * 12) + Credit_History_months
        Debt_to_Income = (Outstanding_Debt / (Annual_income + 0.00001)) * 100

        input_data = {
            'Credit_History_Age': Credit_History_Age,
            'Outstanding_Debt': Outstanding_Debt,
            'Num_Credit_Inquiries': Num_Credit_Inquiries,
            'Delay_from_due_date': Delay_from_due_date,
            'Interest_Rate': Interest_Rate,
            'Num_Bank_Accounts': Num_Bank_Accounts,
            'Debt_to_Income': Debt_to_Income,
            'Payment_of_Min_Amount': Payment_of_Min_Amount,
            'Credit_Mix': Credit_Mix,
        }

        pre = pd.DataFrame([input_data])
        pre['Payment_of_Min_Amount'] = pre['Payment_of_Min_Amount'].map({'Not_mensioned': 0, 'No': 1, 'Yes': 2}).fillna(0)
        pre['Credit_Mix'] = pre['Credit_Mix'].map({'undetermined_credit_mix': 0, 'Good': 1, 'Standard': 2, 'Bad': 3}).fillna(0)

        num_list = [
            'Outstanding_Debt', 'Num_Credit_Inquiries', 'Delay_from_due_date',
            'Interest_Rate', 'Num_Bank_Accounts', 'Debt_to_Income', 'Credit_History_Age'
        ]

        def scale_features(df_input):
            if not scaling_params.empty:
                for i, col in enumerate(num_list):
                    if 'feature' in scaling_params.columns:
                        row = scaling_params[scaling_params['feature'] == col]
                        if not row.empty:
                            c_min = float(row['min'].values[0])
                            c_max = float(row['max'].values[0])
                        else:
                            continue
                    else:
                        c_min = float(scaling_params['min'].iloc[i])
                        c_max = float(scaling_params['max'].iloc[i])

                    denom = c_max - c_min if c_max != c_min else 1.0
                    df_input[col] = (df_input[col] - c_min) / denom
            return df_input

        if st.button("🏦 Predict Credit Score", type="primary", use_container_width=True):
            with st.spinner("Processing input and computing prediction..."):
                scaled_input = scale_features(pre.copy())
                prediction = model.predict(scaled_input)[0]

            st.markdown("---")
            if prediction == "Good":
                st.success(f"### Predicted Result: **{prediction}** 🟢 (Good Credit Rating)")
            elif prediction == "Standard":
                st.warning(f"### Predicted Result: **{prediction}** 🟡 (Standard Credit Rating)")
            else:
                st.error(f"### Predicted Result: **{prediction}** 🔴 (Poor Credit Rating)")
