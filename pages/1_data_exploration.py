import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

st.set_page_config(page_title="Data Exploration", layout="wide")
st.title("🔍 Data Exploration & Visualization")

# Load dataset
dataset_dir = "datasets"
os.makedirs(dataset_dir, exist_ok=True)
available_datasets = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
selected_dataset = st.sidebar.selectbox("Or select a pre-stored dataset", ["None"] + available_datasets)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Uploaded dataset loaded successfully!")
elif selected_dataset != "None":
    df = pd.read_csv(os.path.join(dataset_dir, selected_dataset))
    st.success(f"✅ Pre-stored dataset **{selected_dataset}** loaded successfully!")
else:
    st.warning("⚠️ Please upload or select a dataset.")
    st.stop()

# Dataset Preview
st.write("### 📜 Dataset Preview")
st.dataframe(df.head())

# Data Statistics
st.write("### 📊 Dataset Summary")
st.write(df.describe())

# Missing Values
st.write("### ❗ Missing Values in Dataset")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    st.success("✅ No missing values found in the dataset!")
else:
    st.write(missing_values[missing_values > 0])

# Visualization Options
st.write("### 📈 Data Visualizations")
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

if numeric_cols:
    selected_column = st.selectbox("Select a column for histogram:", numeric_cols)
    
    # Histogram
    fig_hist = px.histogram(df, x=selected_column, nbins=30, title=f"📊 Distribution of {selected_column}")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Box Plot (For Outliers)
    st.write("### 📦 Box Plot (Outlier Detection)")
    fig_box = px.box(df, y=selected_column, title=f"📦 Box Plot of {selected_column}")
    st.plotly_chart(fig_box, use_container_width=True)

    # Pair Plot (Feature Relationships)
    st.write("### 🔗 Pair Plot (Feature Relationships)")
    fig_pair = px.scatter_matrix(df, dimensions=numeric_cols, title="🔗 Pair Plot")
    st.plotly_chart(fig_pair, use_container_width=True)

    # Correlation Heatmap
    st.write("### 🔥 Correlation Heatmap")
    correlation_matrix = df[numeric_cols].corr()
    fig_heatmap = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=list(correlation_matrix.columns),
        y=list(correlation_matrix.index),
        annotation_text=np.round(correlation_matrix.values, 2),
        colorscale="Blues",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

else:
    st.warning("⚠️ No numeric columns available for visualization.")
