import streamlit as st

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ----------------------------------
# Load model and preprocessors
# ----------------------------------

@st.cache_resource
def load_laptop_model():
    return load_model("laptop_price_model.keras")

@st.cache_data
def load_preprocessors():
    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
    
    with open("dropdowns.pkl", "rb") as f:
        dropdowns = pickle.load(f)
        
    # Fit StandardScaler on numerical features from original dataset
    df_raw = pd.read_csv("laptop_price.csv", encoding="latin-1")
    df_raw['Ram'] = df_raw['Ram'].str.replace('GB', '').astype(int)
    df_raw['Weight'] = df_raw['Weight'].str.replace('kg', '').astype(float)
    
    scaler = StandardScaler()
    scaler.fit(df_raw[['Inches', 'Ram', 'Weight']])
    
    # Extract unique screen resolutions
    screen_resolutions = sorted(df_raw["ScreenResolution"].unique())
    
    return model_columns, dropdowns, scaler, screen_resolutions

model = load_laptop_model()
model_columns, dropdowns, scaler, screen_resolutions = load_preprocessors()

# Helper function to get default selectbox index safely
def get_index(lst, val):
    try:
        return lst.index(val)
    except ValueError:
        return 0

# Helper function to convert SSD/HDD to the dataset's Memory string format
def get_memory_string(ssd, hdd):
    if ssd == 0 and hdd == 0:
        return "256GB SSD"
    parts = []
    if ssd > 0:
        if ssd in [1000, 1024]:
            ssd_str = "1TB"
        elif ssd in [2000, 2048]:
            ssd_str = "2TB"
        else:
            ssd_str = f"{ssd}GB"
        parts.append(f"{ssd_str} SSD")
    if hdd > 0:
        if hdd in [1000, 1024]:
            hdd_str = "1TB"
        elif hdd in [2000, 2048]:
            hdd_str = "2TB"
        else:
            hdd_str = f"{hdd}GB"
        parts.append(f"{hdd_str} HDD")
    if len(parts) == 2:
        return f"{parts[0]} +  {parts[1]}"
    elif len(parts) == 1:
        return parts[0]
    return "256GB SSD"

# ----------------------------------
# Streamlit UI
# ----------------------------------

st.title("💻 Laptop Price Prediction")
st.write("Enter laptop specifications to estimate its price.")

# ----------------------------------
# User Inputs
# ----------------------------------
st.subheader("🔧 Laptop Specifications")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Brand", dropdowns["Company"], index=get_index(dropdowns["Company"], "Dell"))
    type_name = st.selectbox("Laptop Type", dropdowns["TypeName"], index=get_index(dropdowns["TypeName"], "Notebook"))
    screen_res = st.selectbox("Screen Resolution", screen_resolutions, index=get_index(screen_resolutions, "Full HD 1920x1080"))
    cpu = st.selectbox("CPU Model", dropdowns["Cpu_brand"], index=get_index(dropdowns["Cpu_brand"], "Intel Core i5 7200U 2.5GHz"))
    gpu = st.selectbox("GPU Model", dropdowns["Gpu_brand"], index=get_index(dropdowns["Gpu_brand"], "Intel HD Graphics 620"))
    os = st.selectbox("Operating System", dropdowns["OpSys"], index=get_index(dropdowns["OpSys"], "Windows 10"))

with col2:
    ram = st.selectbox("RAM (GB)", dropdowns["Ram"], index=get_index(dropdowns["Ram"], 8))
    inches = st.number_input("Screen Size (Inches)", 10.0, 20.0, value=15.6, step=0.1)
    ssd = st.number_input("SSD (GB)", 0, 2000, value=256, step=128)
    hdd = st.number_input("HDD (GB)", 0, 2000, value=0, step=256)
    weight = st.number_input("Weight (kg)", 0.5, 5.0, value=2.0, step=0.1)

# ----------------------------------
# Predict Button
# ----------------------------------
if st.button("🔮 Predict Price"):
    # Fit & apply scaling to numerical features
    scaled_features = scaler.transform([[inches, ram, weight]])[0]
    scaled_inches = scaled_features[0]
    scaled_ram = scaled_features[1]
    scaled_weight = scaled_features[2]

    # Create single-row dataframe aligned with dataset structure
    input_data = {
        "Company": company,
        "TypeName": type_name,
        "Cpu": cpu,
        "Gpu": gpu,
        "OpSys": os,
        "Ram": scaled_ram,
        "Inches": scaled_inches,
        "Memory": get_memory_string(ssd, hdd),
        "ScreenResolution": screen_res,
        "Weight": scaled_weight
    }

    input_df = pd.DataFrame([input_data])

    # One-hot encode categorical features
    encoded_df = pd.get_dummies(input_df)

    # Align columns with training data
    encoded_df = encoded_df.reindex(columns=model_columns, fill_value=0)

    # Prediction
    prediction = model.predict(encoded_df)[0][0]
    
    price_in_euros = prediction
    price_in_rupees = prediction * 90.0  # approximate exchange rate 1 Euro = 90 INR

    # Display price predictions
    st.success(f"💰 Estimated Laptop Price: **₹{int(price_in_rupees):,}** (~€{int(price_in_euros):,})")

    st.caption("⚠️ Prediction is based on historical data and exchange rates, and may vary.")

# ----------------------------------
# Footer
# ----------------------------------

