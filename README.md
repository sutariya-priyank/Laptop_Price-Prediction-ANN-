# Laptop Price Predictor (ANN-based)

This repository features a **Laptop Price Prediction** web application built with **Streamlit** and powered by a deep learning **Artificial Neural Network (ANN)** model. The model estimates the price of a laptop based on its key hardware specifications, providing instant evaluations in both Euros (€) and Indian Rupees (₹).

---

## 🚀 Features

- **Deep Learning Engine**: Utilizes a fully trained Artificial Neural Network (ANN) model built with TensorFlow/Keras to capture complex non-linear relationships in laptop specs and pricing.
- **Interactive UI**: A clean, modern Streamlit dashboard allows users to easily customize and input laptop configurations via simple forms.
- **Automated Data Pipeline**: Seamlessly preprocesses raw inputs by parsing storage configurations (SSD/HDD), encoding categorical columns, and scaling numerical features (RAM, screen size, weight) dynamically using a fitted `StandardScaler`.
- **Robust Feature Alignment**: Employs pickled artifacts to maintain exact training column alignments, preventing errors due to unseen or missing categorical categories.

---

## 🛠️ Tech Stack & Architecture

- **Machine Learning & Deep Learning**: TensorFlow, Keras (Sequential ANN model), Scikit-Learn (`StandardScaler`).
- **Web App Development**: Streamlit.
- **Data Wrangling**: Pandas, NumPy.
- **Serialization**: Pickle (for saving encoding categories, dataset columns, and pipeline configurations).

---

## 📦 Project Structure

- **[app.py](file:///Users/priyank/Desktop/Laptop_Price%20Prediction\(ANN\)/app.py)**: The main entrypoint containing the interactive UI layout, memory string parsing logic, model loading, and predictions.
- **[ANN_Model.ipynb](file:///Users/priyank/Desktop/Laptop_Price%20Prediction\(ANN\)/ANN_Model.ipynb)**: Jupyter Notebook detailing exploratory data analysis, data clean-up, feature engineering, model architecture, training loops, and evaluation.
- **[laptop_price_model.keras](file:///Users/priyank/Desktop/Laptop_Price%20Prediction\(ANN\)/laptop_price_model.keras)**: Trained neural network model file containing saved weights.
- **pkl Files**: Binary pickles storing model columns and drop-down category choices to ensure alignment during inference.

---

## ⚙️ Quick Start

### 1. Install Dependencies
Set up your environment and install the required modules:
```bash
pip install -r requirements.txt
```

### 2. Run the Application
Start the Streamlit dashboard on your local server:
```bash
streamlit run app.py
```
Once run, open the local URL (usually `http://localhost:8501` or `http://localhost:8502`) to explore the app.
