import streamlit as st
import numpy as np
import json
import pickle
import pandas as pd
from collections import defaultdict

# Model ve bilgiler yükleniyor
with open("model.pkl", "rb") as file:
    model, all_symptoms, label_encoder = pickle.load(file)

# CSV'den semptom-hastalık eşleşmelerini çıkar (JSON yerine)
df = pd.read_csv(r"teshis_ornek_arttirilmis_cleaned.csv")  # 👈 dosya yolu uyarlanmalı
symptom_cols = df.columns[1:]

disease_symptom_map = defaultdict(set)
for _, row in df.iterrows():
    disease = row["Disease"]
    for symptom in row[symptom_cols]:
        if pd.notna(symptom):
            disease_symptom_map[disease].add(symptom)

# JSON'dan açıklama, öneri ve uzmanlık bilgileri al
json_path = r"hastalk_acklama.json"
with open(json_path, "r", encoding="utf-8") as f:
    disease_info = json.load(f)

disease_descriptions = {
    entry["Disease"]: {
        "description": entry.get("Description", ""),
        "suggestions": entry.get("Suggestions", []),
        "doctor": entry.get("Doctor", "")
    }
    for entry in disease_info
}

import base64

def get_base64_bg(file_path):
    with open(file_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpg;base64,{encoded}"

bg_img = get_base64_bg(r"arkaplan.gif")  # 👈 klasörün içinde olacak

st.markdown(
    f"""
    <style>
    /* 🔹 Arka plan */
    .stApp {{
        background-image: url("{bg_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        margin: 0;
        padding: 0;
    }}

    /* 🔹 Başlık rengi ve efekti */
    h1 {{
        color: black !important;
        font-weight: bold;
        text-shadow:none;
    }}

    /* 🔹 Giriş alanları arka planı */
    .stTextInput, .stMultiSelect {{
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 8px;
        color: black !important;
    }}
    /* 🔹 Multiselect kutusu: arka planı beyaz, yazı siyah */
        div[data-baseweb="select"] > div {{
        background-color: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
        border-radius: 8px !important;
    }}
    .css-1xc3v61.egzxvld1, .css-1wa3eu0-placeholder {{
        background-color: white !important;
        color: white!important;
    }}
     /* 🖤 Alt yazı ve açıklamalar da siyah */
    .block-container, .stMarkdown, .stText, .stTextInput, .stMultiSelect, .css-1v0mbdj, .css-1v3fvcr {{
        color: black !important;
    }}
    .css-26l3qy-menu {{
        background-color: white !important;
        color: black !important;
    }}  
    .css-12jo7m5 {{
        background-color: white !important;
        color: black !important;
    }}

    /* 🔹 Buton mavi ve net */
    .stButton>button {{
        background-color: #007BFF;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: #0056b3;
        transition: 0.3s;
        color: white !important; 
    }}

    /* 🔹 Ana kutuya yarı saydam fon */
    .block-container {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 12px;
        margin-top: 2rem;
    }}

    /* 🔹 Siyah header arka planını kaldır */
    header {{
        background-color: transparent !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🩺 Yapay Zeka Destekli Hastalık Teşhisi Sistemi")

st.write("Lütfen yaşadığınız semptomları aşağıdan seçiniz:")

selected_symptoms = st.multiselect("Semptomları Seçiniz:", all_symptoms)

if st.button("🔍 Tahmin Et"):

    input_vector = np.zeros(len(all_symptoms))
    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            input_vector[all_symptoms.index(symptom)] = 1

    if np.sum(input_vector) == 0:
        st.warning("⚠️ Lütfen en az bir semptom seçin.")
    else:
        probs = model.predict_proba([input_vector])[0]
        top_indices = np.argsort(probs)[::-1]

        top_diseases = []
        for idx in top_indices:
            if probs[idx] == 0:
                continue

            disease = label_encoder.inverse_transform([model.classes_[idx]])[0]
            top_diseases.append((disease, probs[idx]))

            if len(top_diseases) == 3:
                break

        selected_set = set(selected_symptoms)

        displayed = False
        for disease, _ in top_diseases:
            known_symptoms = disease_symptom_map.get(disease, set())

            # Yalnızca en az 1 ortak semptom varsa göster
            if selected_set & known_symptoms:
                displayed = True
                st.markdown(f"### 🩺 {disease}")
                info = disease_descriptions.get(disease, {})
                st.write(f"**📋 Açıklama:** {info.get('description', 'Açıklama bulunamadı.')}")
                if info.get("suggestions"):
                    st.write("**💡 Öneriler:**")
                    for suggestion in info["suggestions"]:
                        st.markdown(f"- {suggestion}")
                st.write(f"**👨‍⚕️ Önerilen Uzmanlık Alanı:** {info.get('doctor', 'Belirtilmedi.')}")
                st.markdown("---")

        if not displayed:
            st.warning("⚠️ Seçilen semptomlarla eşleşen hastalık bulunamadı.")
