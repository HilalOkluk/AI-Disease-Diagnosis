import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle
from collections import defaultdict

# 1. Veriyi oku
df = pd.read_csv(r"C:\Users\Hilal Öklük\Downloads\teshis_ornek_arttirilmis_cleaned.csv")

# 1. Tüm semptom sütunlarını al (ilk sütun 'Disease')
symptom_cols = df.columns[1:]

# 2. Benzersiz tüm semptomları topla
all_symptoms = sorted(set(symptom for col in symptom_cols for symptom in df[col].dropna().unique()))

# 3. Hastalık başına semptom sıklığını hesapla
disease_symptom_counts = defaultdict(lambda: defaultdict(int))

for _, row in df.iterrows():
    disease = row['Disease']
    for symptom in row[symptom_cols]:
        if pd.notna(symptom):
            disease_symptom_counts[disease][symptom] += 1

# 4. Kritik semptomları elle belirle
critical_symptoms = {
    "bayılma",
    "el uyuşması",
    "kusma",
    "dik durma haricinde nefes darlığı,",
    "bilinç bulanıklığı",
    "kanama",
    "eforla nefes darlığı",
    "kanlı dışkı",
    "idrarda kan",
    "kısmi felç",
    "baygınlık",
    "kanlı idrar",
    "kanlı balgam veya kan tükürme",
    "çarpıntı hissi",
    "çarpıntı",
    "adele ağrısı",
    "bilinçsiz durum",
    "zor nefes alma",
    "mide kanaması",
    "meme ağrısı",
    "kolay kanama veya morarma",
    "kolay morarma ve kanama",
    "nefes almada zorluk",
    "şiddetli karın veya sırt ağrısı",
    "nefessiz kalma",
    "bilinç kaybı",
    "normalden daha hızlı ve derin nefes alma",
    "göğüs ağrısı veya rahatsızlığı",
    "dışkıda kan",
    "anormal vajinal kanama",
    "uyuşma veya karıncalanma",
    "ellerde ve ayaklarda güçsüzlük",
    "kardiyovasküler bulgu",
    "kanlı balgam veya kan tükürme",
    "yumru",
    "şiddetli baş ağrısı"
    "nefes darlığı",
    "şiddetli baş ağrısı",
    "göğüs ağrısı",
    "bilinç kaybı",
    "felç",
    "kan kusma",
}

# 5. Ayırt edici semptomları tespit et (bir hastalıkta baskın olanlar)
distinctive_symptoms = set()
symptom_frequency = defaultdict(lambda: defaultdict(int))

for disease, symptoms in disease_symptom_counts.items():
    for symptom, count in symptoms.items():
        symptom_frequency[symptom][disease] = count

for symptom, disease_counts in symptom_frequency.items():
    if len(disease_counts) == 1:
        distinctive_symptoms.add(symptom)

# 6. Ağırlıkları ata
symptom_weights = {}
for symptom in all_symptoms:
    if symptom in critical_symptoms:
        symptom_weights[symptom] = 3.0
    elif symptom in distinctive_symptoms:
        symptom_weights[symptom] = 2.0
    else:
        symptom_weights[symptom] = 1.0

# Gösterilecek tablo
weights_df = pd.DataFrame({
    "Semptom": list(symptom_weights.keys()),
    "Ağırlık": list(symptom_weights.values())
}).sort_values(by="Ağırlık", ascending=False)

# Binary + ağırlıklı özellik matrisi oluştur
X = pd.DataFrame(0, index=df.index, columns=all_symptoms)
for i, row in df.iterrows():
    for symptom in row[symptom_cols]:
        if pd.notna(symptom) and symptom in X.columns:
            X.at[i, symptom] = symptom_weights.get(symptom, 1.0)  # Ağırlığı uygula

# 5. Hedef değişken
le = LabelEncoder()
y = le.fit_transform(df["Disease"])

# 6. Eğitim/test bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. CV Fold sayısını belirle (tüm sınıflar en az 2 örnek içermeli)
min_class_count = pd.Series(y).value_counts().min()
cv_folds = max(2, min(5, min_class_count))
skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

# 8. Modeller
models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=25, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear")
}

# 9. Eğitim ve Değerlendirme
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    cv_score = cross_val_score(model, X, y, cv=skf).mean()
    results.append((name, test_score, cv_score))
    print(f"✅ {name} - Test Doğruluğu: {test_score * 100:.2f}% - CV Skoru: {cv_score * 100:.2f}%")

# 10. En iyi modeli seç ve kaydet
best_model = max(results, key=lambda x: x[2])[0]
final_model = models[best_model]
with open("model.pkl", "wb") as f:
    pickle.dump((final_model, all_symptoms, le), f)

print(f"\n🏆 Kaydedilen Model: {best_model}")
print("🔢 Toplam Semptom:", len(all_symptoms))
print("🧬 Toplam Hastalık Sayısı:", len(le.classes_))
