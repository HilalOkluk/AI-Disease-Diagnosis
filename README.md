
# 🧠 AI Disease Diagnosis System

This project is an AI-powered web application for disease prediction based on user-entered symptoms. Built using Python, Streamlit, and a trained machine learning model, the system provides fast and accurate disease diagnosis suggestions.

## 🚀 Live Demo

🔗 [Click here to access the live version](https://your-live-link.com)  
> Replace this with your actual deployed app link.

## 💻 Technologies Used

- Python 🐍  
- Streamlit 📊  
- Pandas & Scikit-learn 🧪  
- Machine Learning (Multiclass Classifier)  
- Pickle (for model serialization)  

## 🧬 Features

- Input symptoms via user interface  
- Predict likely disease from trained ML model  
- Clean and user-friendly design  
- Deployed and accessible via web  

## 📦 Installation (For Local Use)

```bash
git clone https://github.com/HilalOkluk/AI-Disease-Diagnosis.git
cd AI-Disease-Diagnosis
pip install -r requirements.txt
streamlit run app.py
```

## 📁 File Structure

- `app.py` – Streamlit-based UI and logic  
- `model.pkl` – Trained machine learning model  
- `symptom_list.pkl` – List of symptoms used for prediction  
- `label_encoder.pkl` – Encoder for disease labels  

## 🧠 Model Info

The machine learning model is trained using a labeled symptom-disease dataset. It uses techniques like Label Encoding and classification algorithms (e.g., Random Forest or similar) to predict the most probable disease.

## 📬 Contact

Created by [Hilal Okluk](https://github.com/HilalOkluk) – feel free to reach out!

---

🛡️ **Note:** This application is for educational purposes only and should not be used for real medical diagnosis.
