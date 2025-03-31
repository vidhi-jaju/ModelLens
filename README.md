# **ModelLens**  
🚀 *A Clear View Into AI Predictions and Accuracy*  
Live Demo - [Click Here](https://modellens.streamlit.app/)

---  

## 📌 **Overview**  
**ModelLens** is a powerful **Streamlit-based** application designed to **interpret machine learning models** using **SHAP (SHapley Additive exPlanations) values**. It provides a **clear and interactive way** to understand how models make predictions by analyzing **feature importance, impact, and accuracy**.  

This project bridges the gap between complex AI models and human interpretability, enabling users to explore **model behavior, visualize feature contributions, and evaluate performance**—all in an intuitive interface.  

---  

## 🎯 **Key Features**  

✔️ **Upload and Analyze ML Models** – Supports `.pkl` model files.  
✔️ **Dataset Selection & Exploration** – Upload your dataset or choose from predefined ones.  
✔️ **Feature Importance Visualization** – Discover which features influence predictions the most.  
✔️ **SHAP Waterfall & Summary Plots** – Understand individual and global feature contributions.  
✔️ **SHAP Dependence Plots** – See how a single feature affects predictions.  
✔️ **Model Accuracy Evaluation** – Calculates accuracy based on user-selected target variables.  
✔️ **Seamless UI & Interactive Graphs** – Built with `Matplotlib`, `Plotly`, and `SHAP`.  

---

## 🛠️ **How It Works?**  

🔹 **Step 1:** Select a trained machine learning model (`.pkl`).  
🔹 **Step 2:** Choose a dataset (`.csv`) to analyze.  
🔹 **Step 3:** Select a **target variable** to evaluate model predictions.  
🔹 **Step 4:** Explore **feature importance** using SHAP summary & waterfall plots.  
🔹 **Step 5:** Visualize **dependence plots** to understand individual feature effects.  
🔹 **Step 6:** Evaluate **model accuracy** to assess its performance.  

---

## 📂 **Project Structure**  

```
ModelLens/
│── models/                 # Folder for storing trained models (.pkl)
│── datasets/               # Folder for dataset files (.csv)
│── app.py                  # Main Streamlit app
│── pages/
│   ├── 1_data_exploration.py  # Data exploration and visualization
│   ├── 2_model_training.py    # Concept & Data Drift analysis
│   ├── 3_evidential_ai.py     # Model training & prediction
│   ├── 4_shapley_values.py    # SHAP-based explainability & accuracy
│── requirements.txt         # Required dependencies
│── README.md                # Project Documentation
```

---

## 🖥️ **Tech Stack**  

🔹 **Frontend:** `Streamlit`, `Matplotlib`, `Plotly`  
🔹 **Backend:** `Python`, `SHAP`, `scikit-learn`, `Pandas`  
🔹 **Storage:** Local `.pkl` models and `.csv` datasets  

---

## 🚀 **Installation & Usage**  

### 🔧 **Setup the Environment**  
```bash
git clone https://github.com/your-username/ModelLens.git
cd ModelLens
pip install -r requirements.txt
```

### ▶ **Run the Application**  
```bash
streamlit run app.py
```

### 📝 **Upload and Analyze Models**  
- Place `.pkl` models in the `models/` folder.  
- Place `.csv` datasets in the `datasets/` folder.  
- Select them from the **left sidebar** in the Streamlit app.  

---

## 📈 **Visualizations & Insights**  

### 🔍 **Feature Importance - SHAP Summary Plot**  
Understand which features impact predictions the most.  

### 🔹 **Waterfall Plot**  
Shows how individual features contribute to a single prediction.  

### 📊 **Dependence Plot**  
Examines how a specific feature interacts with model predictions.  

### ✅ **Model Accuracy**  
Evaluates how well the model performs on the selected dataset.  

---

## 🤖 **Why Use ModelLens?**  

🔍 **Improve Model Transparency** – Understand AI decision-making.  
📊 **Interactive Visualizations** – SHAP-powered insights at your fingertips.  
📈 **Assess Model Performance** – Know how accurate your models are.  
⚡ **User-Friendly Interface** – No coding required, just upload and analyze!  

---
![img](https://github.com/vidhi-jaju/ModelLens/blob/bd326adab6a8010541f7f263e776a24237090f4c/images/1.png)

![img](https://github.com/vidhi-jaju/ModelLens/blob/bd326adab6a8010541f7f263e776a24237090f4c/images/2.png)

![img](https://github.com/vidhi-jaju/ModelLens/blob/bd326adab6a8010541f7f263e776a24237090f4c/images/3.png)

![img](https://github.com/vidhi-jaju/ModelLens/blob/bd326adab6a8010541f7f263e776a24237090f4c/images/4.png)

![img](https://github.com/vidhi-jaju/ModelLens/blob/bd326adab6a8010541f7f263e776a24237090f4c/images/5.png)

![img](https://github.com/vidhi-jaju/ModelLens/blob/bd326adab6a8010541f7f263e776a24237090f4c/images/6.png)

## **Deployed on streamlit community cloud** 
![img](https://github.com/vidhi-jaju/ModelLens/blob/bd326adab6a8010541f7f263e776a24237090f4c/images/7.png)

🚀 **Explore, Explain, and Evaluate AI with ModelLens!** 🎯


