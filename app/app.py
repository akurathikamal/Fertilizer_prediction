import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os


MODEL_PATH = os.path.join("..", "model", "fertilizer_model.pkl")
ENCODERS_PATH = os.path.join("..", "model", "encoders.pkl")
DATA_PATH = os.path.join("..", "data", "fertilizer_data.csv")


try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    encoders = pickle.load(open(ENCODERS_PATH, "rb"))
except Exception as e:
    st.error(f"Could not load model or encoders. Check files at {MODEL_PATH} and {ENCODERS_PATH}. Error: {e}")
    st.stop()

le_soil = encoders.get("soil")
le_crop = encoders.get("crop")
le_fert = encoders.get("fert")


try:
    df = pd.read_csv(DATA_PATH, header=0)
except Exception as e:
    st.error(f"Could not load dataset at {DATA_PATH}. Error: {e}")
    st.stop()



df.columns = [c.strip() for c in df.columns]


fix_map = {
    "Temparature": "Temperature",
    "Temperature ": "Temperature",
    "Humidity ": "Humidity",
    "Soil_Type": "Soil Type",
    "Crop_Type": "Crop Type",
    "Fertilizer_Name": "Fertilizer Name"
}
df = df.rename(columns={k: v for k, v in fix_map.items() if k in df.columns})




required_numeric = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
required_categorical = ["Soil Type", "Crop Type"]
required_all = required_numeric + required_categorical

missing = [c for c in required_all if c not in df.columns]
if missing:
    st.error(f"Dataset is missing required columns: {missing}. Actual columns: {df.columns.tolist()}")
    st.stop()


for col in required_numeric:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if df[required_numeric].isnull().all(axis=None):
    st.error("Numeric columns contain no valid numeric data. Check your CSV.")
    st.stop()


ranges = {}
for col in required_numeric:
    series = df[col].dropna()
    ranges[col] = {
        "min": float(series.min()),
        "q1": float(series.quantile(0.25)),
        "q2": float(series.quantile(0.50)),
        "q3": float(series.quantile(0.75)),
        "max": float(series.max())
    }

def classify_value(val, col):
    r = ranges[col]
    if val <= r["q1"]:
        return "Low"
    elif val <= r["q2"]:
        return "Medium-Low"
    elif val <= r["q3"]:
        return "Medium-High"
    else:
        return "High"

def explain_prediction(sample, fert_name):
    temp, humidity, moisture, soil_enc, crop_enc, n, k, p = sample[0]


    soil_name = None
    crop_name = None
    try:
        if le_soil is not None:
            soil_name = le_soil.inverse_transform([int(soil_enc)])[0]
    except Exception:
        soil_name = str(soil_enc)
    try:
        if le_crop is not None:
            crop_name = le_crop.inverse_transform([int(crop_enc)])[0]
    except Exception:
        crop_name = str(crop_enc)

    if soil_name is None:
        soil_name = str(soil_enc)
    if crop_name is None:
        crop_name = str(crop_enc)

    lines = []
    lines.append("### 🔍 Why this fertilizer was recommended\n")

    lines.append(f"**1. Temperature = {temp}°C** → {classify_value(temp, 'Temperature')} range (quartiles: {round(ranges['Temperature']['q1'],1)}, {round(ranges['Temperature']['q2'],1)}, {round(ranges['Temperature']['q3'],1)}).")
    lines.append("Common fertilizers in this temperature range: Urea, DAP, 28-28.")

    lines.append(f"\n**2. Humidity = {humidity}%** → {classify_value(humidity, 'Humidity')} range (quartiles: {round(ranges['Humidity']['q1'],1)}, {round(ranges['Humidity']['q2'],1)}, {round(ranges['Humidity']['q3'],1)}).")
    lines.append("Common fertilizers in this humidity range: Urea, 14-35-14.")

    lines.append(f"\n**3. Moisture = {moisture}%** → {classify_value(moisture, 'Moisture')} range (quartiles: {round(ranges['Moisture']['q1'],1)}, {round(ranges['Moisture']['q2'],1)}, {round(ranges['Moisture']['q3'],1)}).")
    lines.append("Common fertilizers in this moisture range: Urea, DAP.")

    lines.append(f"\n**4. Soil Type = {soil_name}** → soil texture and retention affect fertilizer choice; commonly recommended: Urea, DAP.")
    lines.append(f"\n**5. Crop Type = {crop_name}** → crop nutrient needs often favor nitrogen-rich fertilizers like Urea.")

    lines.append(f"\n**6. Nitrogen = {n}** → {classify_value(n, 'Nitrogen')} range (quartiles: {round(ranges['Nitrogen']['q1'],1)}, {round(ranges['Nitrogen']['q2'],1)}, {round(ranges['Nitrogen']['q3'],1)}).")
    lines.append("Nitrogen-rich fertilizers such as Urea match medium/high nitrogen needs.")

    lines.append(f"\n**7. Potassium = {k}** → {classify_value(k, 'Potassium')} range (quartiles: {round(ranges['Potassium']['q1'],1)}, {round(ranges['Potassium']['q2'],1)}, {round(ranges['Potassium']['q3'],1)}).")
    lines.append("Potassium requirement low/medium → balanced blends or Urea may be suitable depending on crop.")

    lines.append(f"\n**8. Phosphorous = {p}** → {classify_value(p, 'Phosphorous')} range (quartiles: {round(ranges['Phosphorous']['q1'],1)}, {round(ranges['Phosphorous']['q2'],1)}, {round(ranges['Phosphorous']['q3'],1)}).")
    lines.append("Phosphorous requirement low/medium → DAP or blends may be considered if P is low.")

    lines.append(f"\n### 🎯 Final Conclusion\nBased on all conditions above, the recommended fertilizer is **{fert_name}**.")

    return "\n\n".join(lines)

st.title("🌾 Fertilizer Recommendation System")

st.markdown("Enter field conditions and click Predict. Ranges are computed from your dataset.")


temp = st.number_input("Temperature", min_value=float(ranges["Temperature"]["min"]), max_value=float(ranges["Temperature"]["max"]), value=float(ranges["Temperature"]["q2"]))
humidity = st.number_input("Humidity", min_value=float(ranges["Humidity"]["min"]), max_value=float(ranges["Humidity"]["max"]), value=float(ranges["Humidity"]["q2"]))
moisture = st.number_input("Moisture", min_value=float(ranges["Moisture"]["min"]), max_value=float(ranges["Moisture"]["max"]), value=float(ranges["Moisture"]["q2"]))

try:
    soil_options = list(le_soil.classes_)
except Exception:
    soil_options = df["Soil Type"].astype(str).unique().tolist()

try:
    crop_options = list(le_crop.classes_)
except Exception:
    crop_options = df["Crop Type"].astype(str).unique().tolist()

soil = st.selectbox("Soil Type", soil_options)
crop = st.selectbox("Crop Type", crop_options)

nitrogen = st.number_input("Nitrogen", min_value=float(ranges["Nitrogen"]["min"]), max_value=float(ranges["Nitrogen"]["max"]), value=float(ranges["Nitrogen"]["q2"]))
potassium = st.number_input("Potassium", min_value=float(ranges["Potassium"]["min"]), max_value=float(ranges["Potassium"]["max"]), value=float(ranges["Potassium"]["q2"]))
phosphorous = st.number_input("Phosphorous", min_value=float(ranges["Phosphorous"]["min"]), max_value=float(ranges["Phosphorous"]["max"]), value=float(ranges["Phosphorous"]["q2"]))


if st.button("Predict Fertilizer"):
    try:
        soil_enc = le_soil.transform([soil])[0]
    except Exception:
        try:
            soil_enc = int(soil)
        except Exception:
            soil_enc = soil

    try:
        crop_enc = le_crop.transform([crop])[0]
    except Exception:
        try:
            crop_enc = int(crop)
        except Exception:
            crop_enc = crop

    sample = np.array([[temp, humidity, moisture, soil_enc, crop_enc, nitrogen, potassium, phosphorous]])

    
    try:
        prediction = model.predict(sample)
        fert_name = le_fert.inverse_transform(prediction)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.success(f"🌱 Recommended Fertilizer: **{fert_name}**")

    explanation = explain_prediction(sample, fert_name)
    st.markdown(explanation)