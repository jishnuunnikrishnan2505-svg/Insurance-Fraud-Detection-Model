import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🚨",
    layout="wide"
)

@st.cache_resource
def load_assets():
    model = joblib.load("insurance_fraud_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    train_cols = joblib.load("train_feature_columns.pkl")
    return model, encoders, train_cols

model, label_encoders, train_cols = load_assets()

st.title("🚨 Insurance Claim Fraud Detection")
st.markdown("""
This application predicts whether an insurance claim is **Fraudulent or Genuine**
using a **Machine Learning model**.
""")

uploaded_file = st.file_uploader(
    "Upload Insurance Claims File (CSV / Excel / JSON)",
    type=["csv", "xlsx", "xls", "json"]
)


def preprocess_data(df, train_cols, encoders):
    df_processed = df.copy()

    # Drop same columns as notebook
    cols_to_drop = [
        'policy_number',
        'policy_bind_date',
        'incident_date',
        'incident_location',
        'incident_city',
        'incident_state',
        'auto_make',
        'auto_model'
    ]

    df_processed.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Fill missing values
    num_cols = df_processed.select_dtypes(include=['int64','float64']).columns
    df_processed[num_cols] = df_processed[num_cols].fillna(
        df_processed[num_cols].median()
    )

    cat_cols = df_processed.select_dtypes(include='object').columns
    df_processed[cat_cols] = df_processed[cat_cols].fillna(
        df_processed[cat_cols].mode().iloc[0]
    )

    # Encode categorical columns
    for col in cat_cols:
        if col in encoders:
            df_processed[col] = encoders[col].transform(df_processed[col])

    # Ensure column order matches training
    df_processed = df_processed.reindex(columns=train_cols, fill_value=0)

    return df_processed

def load_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            return pd.read_csv(uploaded_file)

        elif file_name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)

        elif file_name.endswith(".json"):
            return pd.read_json(uploaded_file)

        else:
            st.error("Unsupported file format.")
            return None

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

if uploaded_file is not None:
    df_input = load_uploaded_file(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df_input.head())

    processed_data = preprocess_data(
        df_input, train_cols, label_encoders
    )

    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)[:, 1]

    df_input['Fraud_Prediction'] = np.where(
        predictions == 1, "Fraud", "Genuine"
    )
    df_input['Fraud_Probability'] = probabilities.round(2)

    st.subheader("🔍 Prediction Results")
    st.dataframe(df_input)

    st.download_button(
        "Download Predictions CSV",
        df_input.to_csv(index=False),
        "fraud_predictions.csv",
        "text/csv"
    )

with st.expander("ℹ Required Dataset Format"):
    st.markdown("""
- File formats supported: **CSV, Excel (.xlsx/.xls), JSON**
- File must contain **same columns as training data**
""")



# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # --------------------------------------------------
# # Page Configuration
# # --------------------------------------------------
# st.set_page_config(
#     page_title="Insurance Fraud Detection",
#     page_icon="🚨",
#     layout="wide"
# )

# # --------------------------------------------------
# # Custom CSS (UI ONLY – no functional impact)
# # --------------------------------------------------
# st.markdown("""
# <style>
# .main {
#     background-color: #f8f9fa;
# }
# .block-container {
#     padding-top: 2rem;
# }
# h1, h2, h3 {
#     color: #1f2937;
# }
# .metric-container {
#     background: white;
#     padding: 1.2rem;
#     border-radius: 12px;
#     box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
#     text-align: center;
# }
# .footer {
#     text-align: center;
#     color: #6b7280;
#     font-size: 0.85rem;
#     margin-top: 2rem;
# }
# .upload-box {
#     background: white;
#     padding: 1.5rem;
#     border-radius: 12px;
#     box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
# }
# </style>
# """, unsafe_allow_html=True)

# # --------------------------------------------------
# # Load Assets
# # --------------------------------------------------
# @st.cache_resource
# def load_assets():
#     model = joblib.load("insurance_fraud_model.pkl")
#     encoders = joblib.load("label_encoders.pkl")
#     train_cols = joblib.load("train_feature_columns.pkl")
#     return model, encoders, train_cols

# model, label_encoders, train_cols = load_assets()

# # --------------------------------------------------
# # Header Section
# # --------------------------------------------------
# st.markdown("## 🚨 Insurance Claim Fraud Detection System")
# st.markdown(
#     "A **machine learning powered web application** to identify "
#     "**fraudulent vs genuine insurance claims** with confidence scores."
# )

# st.divider()

# # --------------------------------------------------
# # File Upload Section
# # --------------------------------------------------
# with st.container():
#     st.markdown("### 📤 Upload Insurance Claims Data")
#     st.markdown(
#         "Upload your dataset to instantly detect **potentially fraudulent claims**."
#     )

#     uploaded_file = st.file_uploader(
#         "Supported formats: CSV, Excel, JSON",
#         type=["csv", "xlsx", "xls", "json"]
#     )

# # --------------------------------------------------
# # Data Preprocessing
# # --------------------------------------------------
# def preprocess_data(df, train_cols, encoders):
#     df_processed = df.copy()

#     cols_to_drop = [
#         'policy_number',
#         'policy_bind_date',
#         'incident_date',
#         'incident_location',
#         'incident_city',
#         'incident_state',
#         'auto_make',
#         'auto_model'
#     ]

#     df_processed.drop(columns=cols_to_drop, inplace=True, errors='ignore')

#     num_cols = df_processed.select_dtypes(include=['int64','float64']).columns
#     df_processed[num_cols] = df_processed[num_cols].fillna(
#         df_processed[num_cols].median()
#     )

#     cat_cols = df_processed.select_dtypes(include='object').columns
#     df_processed[cat_cols] = df_processed[cat_cols].fillna(
#         df_processed[cat_cols].mode().iloc[0]
#     )

#     for col in cat_cols:
#         if col in encoders:
#             df_processed[col] = encoders[col].transform(df_processed[col])

#     df_processed = df_processed.reindex(columns=train_cols, fill_value=0)
#     return df_processed

# # --------------------------------------------------
# # Load Uploaded File
# # --------------------------------------------------
# def load_uploaded_file(uploaded_file):
#     file_name = uploaded_file.name.lower()
#     try:
#         if file_name.endswith(".csv"):
#             return pd.read_csv(uploaded_file)
#         elif file_name.endswith((".xlsx", ".xls")):
#             return pd.read_excel(uploaded_file)
#         elif file_name.endswith(".json"):
#             return pd.read_json(uploaded_file)
#         else:
#             st.error("Unsupported file format.")
#             return None
#     except Exception as e:
#         st.error(f"Error reading file: {e}")
#         return None

# # --------------------------------------------------
# # Prediction Pipeline
# # --------------------------------------------------
# if uploaded_file is not None:
#     df_input = load_uploaded_file(uploaded_file)

#     if df_input is not None:
#         st.divider()

#         # Preview Section
#         st.markdown("### 📄 Uploaded Data Preview")
#         st.dataframe(df_input.head(), use_container_width=True)

#         processed_data = preprocess_data(
#             df_input, train_cols, label_encoders
#         )

#         predictions = model.predict(processed_data)
#         probabilities = model.predict_proba(processed_data)[:, 1]

#         df_input['Fraud_Prediction'] = np.where(
#             predictions == 1, "Fraud", "Genuine"
#         )
#         df_input['Fraud_Probability'] = probabilities.round(2)

#         st.divider()

#         # Metrics Section
#         fraud_count = (df_input['Fraud_Prediction'] == "Fraud").sum()
#         genuine_count = (df_input['Fraud_Prediction'] == "Genuine").sum()

#         col1, col2, col3 = st.columns(3)

#         with col1:
#             st.metric("Total Claims", len(df_input))
#         with col2:
#             st.metric("Fraudulent Claims", fraud_count)
#         with col3:
#             st.metric("Genuine Claims", genuine_count)

#         st.divider()

#         # Results Section
#         st.markdown("### 🔍 Prediction Results")
#         st.dataframe(df_input, use_container_width=True)

#         st.download_button(
#             "⬇ Download Predictions CSV",
#             df_input.to_csv(index=False),
#             "fraud_predictions.csv",
#             "text/csv"
#         )

# # --------------------------------------------------
# # Dataset Info Section
# # --------------------------------------------------
# with st.expander("ℹ Required Dataset Format"):
#     st.markdown("""
#     **Guidelines:**
#     - Supported formats: **CSV, Excel (.xlsx/.xls), JSON**
#     - Dataset must contain **same columns used during model training**
#     - Missing values are handled automatically
#     - Output includes **Fraud label + probability score**
#     """)

# # --------------------------------------------------
# # Footer
# # --------------------------------------------------
# st.markdown("""
# <div class="footer">
#     🔐 Insurance Fraud Detection System • Built with Streamlit & Machine Learning
# </div>
# """, unsafe_allow_html=True)
