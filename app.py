import streamlit as st
import os
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("🚀 Machine Learning Dashboard")

st.sidebar.header("📂 Upload or Select a Dataset")

# Ensure directories exist
dataset_dir = "datasets"
model_dir = "models"
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# List available datasets
available_datasets = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Dataset selection
selected_dataset = st.sidebar.selectbox("Or select a pre-stored dataset", ["None"] + available_datasets)

# Button to clear selection
if st.sidebar.button("Clear Selection"):
    uploaded_file = None
    selected_dataset = "None"
    st.experimental_rerun()

# Load dataset
df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
        st.success("✅ Uploaded dataset loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
elif selected_dataset != "None":
    try:
        df = pd.read_csv(os.path.join(dataset_dir, selected_dataset), encoding="utf-8")
        st.success(f"✅ Pre-stored dataset **{selected_dataset}** loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")

# Display dataset preview
if df is not None:
    st.write("### 📊 Dataset Preview")
    st.dataframe(df, height=400)  # Added height for better scrolling

    # Save dataset info
    if st.button("💾 Save Dataset Info"):
        dataset_info = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "columns_list": df.columns.tolist()
        }
        dataset_info_path = os.path.join(dataset_dir, "dataset_info.pkl")
        joblib.dump(dataset_info, dataset_info_path)
        st.success("✅ Dataset info saved successfully!")

    # Allow users to save a trained model
    st.write("### 💾 Save Model")
    model_name = st.text_input("Enter model name (e.g., my_model.pkl)", "trained_model.pkl")
    if st.button("Save Model"):
        model_path = os.path.join(model_dir, model_name)
        try:
            joblib.dump(df, model_path)  # Here, replace `df` with your trained model
            st.success(f"✅ Model saved successfully as `{model_name}`!")
        except Exception as e:
            st.error(f"❌ Error saving model: {e}")
