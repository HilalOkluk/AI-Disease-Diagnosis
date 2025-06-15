# 🧠 Yapay Zeka Destekli Hastalık Teşhisi Sistemi

Bu proje, kullanıcının seçtiği semptomlara göre en olası hastalıkları tahmin eden bir yapay zeka sistemidir. Arayüz, **Streamlit** ile hazırlanmıştır. Model, kullanıcıya tahmin edilen hastalığın açıklamasını, önerilerini ve ilgili tıbbi bölümü sunar.

---

## 🚀 Özellikler

- Semptom seçimine göre olasılık tabanlı hastalık tahmini
- En yüksek olasılıklı 3 hastalık listesi
- Her hastalık için:
  - 📋 Açıklama
  - 💡 Tedavi/önlem önerileri
  - 👨‍⚕️ İlgili uzmanlık alanı
- Eğitilmiş makine öğrenmesi modeli (Random Forest)
- Gelişmiş semptom ağırlıklandırma: kritik, ayırt edici ve standart
- Arka plan görselli, kullanıcı dostu Streamlit arayüzü

---

## 🗂️ Dosya Yapısı

```
AI-Disease-Diagnosis/
├── app.py                        # Streamlit arayüzü
├── train_model.py               # Model eğitimi ve ağırlıklandırma
├── model.pkl                    # Eğitilmiş model + semptom listesi + LabelEncoder
├── teshis_ornek_arttirilmis_cleaned.csv   # Eğitim veri kümesi
├── hastalk_acklama.json         # Hastalık açıklamaları, öneriler, uzmanlıklar
├── ad49d6a3.gif                 # Arka plan görseli
└── README.md                    # Proje tanımı
```

---

## ⚙️ Kurulum

### 📦 Gerekli Python Paketleri

```bash
pip install streamlit scikit-learn pandas numpy
```

> `json`, `pickle`, `collections` gibi modüller Python ile birlikte gelir; ayrıca kurulum gerektirmez.

---

## 💻 Lokal Çalıştırma (Localhost)

Proje klasörüne terminal/komut satırı ile gidin ve aşağıdaki komutu çalıştırın:

```bash
streamlit run app.py
```

Ardından otomatik olarak şu adres açılır:

```
http://localhost:8501
```

---

## 🌐 Uygulamayı İnternete Açmak (Ücretsiz Yayın)

Projeni çevrim içi olarak başkalarının da erişebileceği şekilde yayınlamak istersen, **Streamlit Cloud** kullanabilirsin:

### 🔹 Yayınlama Adımları:

1. Projeni GitHub’a yükle (örn: `https://github.com/kullaniciadi/AI-Disease-Diagnosis`)
2. [https://streamlit.io/cloud](https://streamlit.io/cloud) adresine git
3. “Sign in with GitHub” diyerek giriş yap
4. `AI-Disease-Diagnosis` reposunu seç
5. Uygulama dosyası olarak `app.py` seçili olduğundan emin ol
6. “Deploy” butonuna bas

✅ Projen 1-2 dakika içinde çalışır ve sana özel bir bağlantı verilir:  
Örn: `https://hilalokluk-ai-diagnosis.streamlit.app`

---

## 🧪 Ek Bilgiler

- `model.pkl`, `train_model.py` ile eğitilerek oluşturulmalıdır.
- `hastalk_acklama.json`, her hastalık için açıklama, öneri ve uzmanlık alanı içerir.
- `ad49d6a3.gif`, arka plan olarak projede otomatik olarak çağrılır. Aynı dizinde olmalıdır.

---

## 👩‍💻 Geliştirici

**Hilal Öklük**  
📅 2025 – Bitirme Projesi  
🧠 Python • Streamlit • Makine Öğrenmesi • Yapay Zeka

---