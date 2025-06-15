
# 🧠 Yapay Zekâ Destekli Hastalık Teşhis Sistemi

Bu proje, kullanıcı tarafından girilen semptomlara dayanarak hastalık tahmini yapan yapay zekâ destekli bir web uygulamasıdır. Python, Streamlit ve eğitilmiş bir makine öğrenmesi modeli kullanılarak geliştirilmiştir. Sistem hızlı ve doğru teşhis önerileri sunar.

## 🚀 Canlı Uygulama

🔗 [Canlı Uygulamayı Görüntüle](https://4dscxhwqqq8nlqlrgaux53.streamlit.app/)

## 💻 Kullanılan Teknolojiler

- Python 🐍  
- Streamlit 📊  
- Pandas & Scikit-learn 🧪  
- Makine Öğrenmesi (Çok sınıflı sınıflandırma)  
- Pickle (Model kaydetme ve yükleme)  

## 🧬 Özellikler

- Kullanıcı arayüzü üzerinden semptom girişi  
- Eğitilmiş model ile hastalık tahmini  
- Temiz ve kullanıcı dostu tasarım  
- Web üzerinden erişilebilirlik  

## 📦 Kurulum (Yerel Kullanım İçin)

```bash
git clone https://github.com/HilalOkluk/AI-Disease-Diagnosis.git
cd AI-Disease-Diagnosis
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Dosya Yapısı

- `app.py` – Streamlit tabanlı kullanıcı arayüzü ve tahmin mantığı  
- `model.pkl` – Eğitilmiş makine öğrenmesi modeli  
- `symptom_list.pkl` – Kullanılabilir semptom listesi  
- `label_encoder.pkl` – Hastalık isimlerini kodlayan etiketleyici  

## 🧠 Model Bilgisi

Makine öğrenmesi modeli, semptom-hastalık etiketli bir veri kümesi ile eğitilmiştir. Etiket kodlama ve sınıflandırma algoritmaları (örneğin Random Forest) kullanılarak en olası hastalık tahmini yapılır.

## 📬 İletişim

Geliştirici: [Hilal Okluk](https://github.com/HilalOkluk)

---

🛡️ **Not:** Bu uygulama yalnızca eğitim amaçlıdır ve gerçek tıbbi teşhis yerine geçmez.
