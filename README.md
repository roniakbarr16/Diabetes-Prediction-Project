# 🏥 Diabetes Prediction System

**UAS Pembelajaran Mesin — Genap 2025/2026**  
Prediksi Diabetes Menggunakan Machine Learning (Naive Bayes & Decision Tree)

[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)](https://diabetes-prediction-app.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)](https://scikit-learn.org/)

---

## 📋 Gambaran Proyek

Proyek ini mengembangkan **sistem prediksi diabetes** menggunakan algoritma **Naive Bayes** dan **Decision Tree** untuk membandingkan karakteristik algoritma klasifikasi berdasarkan kompleksitas, akurasi, dan interpretabilitas model. Sistem ini di-deploy sebagai aplikasi web interaktif menggunakan **Streamlit**.

### 🎯 Tujuan
- Mendeteksi risiko diabetes secara dini menggunakan machine learning
- Menyediakan alat bantu skrining awal yang mudah diakses
- Memberikan interpretasi hasil yang dapat dipahami oleh non-teknis

---

## 📊 Dataset

**PIMA Indian Diabetes Dataset** — Sumber: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) / [UCI Repository](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes)

- **768 sampel** perempuan keturunan Indian PIMA
- **8 fitur** + 1 target biner (Outcome)
- **Class distribution:** 65.1% Non-Diabetes, 34.9% Diabetes

| Fitur | Deskripsi |
|-------|-----------|
| Pregnancies | Jumlah kehamilan |
| Glucose | Kadar glukosa plasma (mg/dL) |
| BloodPressure | Tekanan darah diastolik (mm Hg) |
| SkinThickness | Tebal lipatan kulit triceps (mm) |
| Insulin | Kadar insulin serum (μU/ml) |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Riwayat diabetes keluarga |
| Age | Usia (tahun) |

---

## 🛠️ Pipeline Machine Learning

```
Data Acquisition → EDA → Preprocessing → Feature Engineering → Modeling → Tuning → Evaluation → Deployment
```

### Teknik yang Digunakan
- **Handling Missing Values:** Imputasi median (nilai 0 → NaN → median)
- **Feature Engineering:** Kategori BMI/Glukosa/Usia, interaksi fitur
- **Outlier Handling:** Winsorization (1-99 percentile)
- **Scaling:** StandardScaler (Z-score normalization)
- **Class Imbalance:** SMOTE (Synthetic Minority Oversampling)
- **Hyperparameter Tuning:** GridSearchCV dengan 5-fold cross-validation

---

## 🧪 Model & Performa

Dua algoritma dibandingkan (Sesuai Sub-CPMK 8.1.2 yang membandingkan KNN, Decision Tree, dan Naive Bayes):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Naive Bayes** 🏆 | **0.75** | **0.61** | **0.80** | **0.69** | **0.84** |
| Decision Tree | 0.66 | 0.52 | 0.80 | 0.63 | 0.74 |

**Model Terbaik: Gaussian Naive Bayes** — dipilih berdasarkan F1-Score dan ROC-AUC tertinggi. Naive Bayes unggul karena:
- Recall tinggi (0.80) — mampu mendeteksi 80% pasien diabetes
- ROC-AUC tinggi (0.84) — kemampuan diskriminasi yang baik
- Lebih sederhana dan cepat dibanding Decision Tree
- Tidak mudah overfitting pada dataset kecil

---

## 🚀 Cara Menjalankan

### Lokal
```bash
# Clone repository
git clone https://github.com/username/diabetes-prediction.git
cd diabetes-prediction

# Install dependencies
pip install -r requirements.txt

# Jalankan Streamlit
streamlit run app/streamlit_app.py
```

### Deployment
Aplikasi telah di-deploy di Streamlit Cloud:  
🔗 **[Diabetes Prediction App](https://diabetes-prediction-project-machine-learning.streamlit.app/)**

---

## 📁 Struktur Proyek

```
diabetes-prediction-project/
├── data/
│   └── diabetes.csv                    # Dataset PIMA Indian Diabetes
├── notebooks/
│   └── UAS_ML_Diabetes_Prediction.ipynb # EDA + Preprocessing + Modeling
├── models/
│   ├── diabetes_model.pkl               # Model Logistic Regression
│   ├── scaler.pkl                       # StandardScaler
│   ├── imputer.pkl                      # Median Imputer
│   └── feature_names.pkl                # Feature names
├── app/
│   └── streamlit_app.py                 # Streamlit web app
├── assets/
│   └── *.png                            # Visualizations
├── docs/
│   └── laporan.pdf                      # Laporan teknis
├── requirements.txt                     # Python dependencies
└── README.md                            # Dokumentasi
```

---

## 🌐 Halaman Aplikasi Streamlit

1. **📊 Dashboard EDA** — Visualisasi interaktif eksplorasi data
2. **🔬 Model Demo & Prediksi** — Input data → prediksi real-time
3. **📈 Evaluasi Model** — Perbandingan performa dan metrik
4. **💡 Interpretasi Hasil** — Insights bisnis dan rekomendasi
5. **📖 Dokumentasi** — Informasi lengkap proyek

---

## 👨‍💻 Teknologi

| Kategori | Tools |
|----------|-------|
| Bahasa | Python 3.14 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, Imbalanced-learn |
| Visualisasi | Matplotlib, Seaborn, Plotly |
| Deployment | Streamlit |
| Version Control | Git, GitHub |

---

## 📚 Referensi

1. Smith, J.W., et al. (1988). *Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus.*
2. UCI Machine Learning Repository — PIMA Indians Diabetes Dataset
3. WHO — [Diabetes Fact Sheet](https://www.who.int/news-room/fact-sheets/detail/diabetes)
4. Scikit-learn Documentation — https://scikit-learn.org/
5. Streamlit Documentation — https://docs.streamlit.io/

---

*Dibuat untuk memenuhi Ujian Akhir Semester Mata Kuliah Pembelajaran Mesin — Genap 2025/2026*
