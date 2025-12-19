# 🌾 Fertilizer Recommendation System

The **Fertilizer Recommendation System** is a machine learning application that predicts the most suitable fertilizer based on environmental and soil conditions. Built with **Python** and **Streamlit**, it uses a **Random Forest model** to deliver accurate recommendations along with clear, data‑driven explanations for each input.

🔗 **Live Demo:** [Fertilizer Prediction App](https://fertilizerprediction-hyudi6fttzkfmpotxextsu.streamlit.app/)

---

## 📌 Overview
This project analyzes parameters such as Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, and Phosphorous to recommend the optimal fertilizer and explain the reasoning behind the choice.

---

## ✨ Features
- Random Forest–based prediction model  
- Interactive Streamlit web interface  
- Human‑readable explanations for each feature  
- Dataset‑driven range classification (Low / Medium / High)  
- Easy deployment on Streamlit Cloud  

---

## 🛠️ Tech Stack
- Python 3.13+  
- Streamlit  
- Pandas  
- Scikit‑learn  
- Pickle  

---


## 📂 Project Structure

<pre>
fertilizer_project/
├── app/
│   └── app.py                  # Streamlit application
├── data/
│   └── fertilizer_data.csv     # Dataset
├── model/
│   ├── train_model.py          # Training script
│   ├── fertilizer_model.pkl    # Trained Random Forest model
│   └── encoders.pkl            # Label encoders
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
</pre>


---
<pre>
## 🚀 Getting Started
1. Clone the repository:
   git clone https://github.com/akurathikamal/Fertilizer_prediction.git
   cd fertilizer_project/app
2. Install dependencies:
   pip install -r requirements.txt
3. Run the application:
   streamlit run app.py
</pre>
