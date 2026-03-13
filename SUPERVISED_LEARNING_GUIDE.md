<div align="center">

# 🤖 Panduan Lengkap Supervised Machine Learning

**Dari Data Mentah hingga Model Siap Deploy**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ready-189ab4?style=flat-square)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<br/>

> *"Supervised learning adalah fondasi dari sebagian besar aplikasi ML di industri saat ini —*
> *memahaminya dengan benar adalah investasi terpenting seorang data scientist."*

<br/>

```
Definisi Masalah → EDA → Preprocessing → Feature Engineering
→ Split Data → Pilih Algoritma → Evaluasi → Tuning → Deploy
```

</div>

---

## 📋 Daftar Isi

| # | Tahap | Topik Utama |
|---|-------|-------------|
| [1](#-tahap-1-definisi-masalah) | **Definisi Masalah** | Problem statement, tipe task, metrik, baseline |
| [2](#-tahap-2-pengumpulan-data--eda) | **EDA** | Eksplorasi, distribusi, korelasi, outlier |
| [3](#-tahap-3-data-preprocessing) | **Preprocessing** | Missing values, encoding, scaling |
| [4](#-tahap-4-feature-engineering) | **Feature Engineering** | Fitur baru, seleksi, dimensionality reduction |
| [5](#-tahap-5-pembagian-data) | **Train/Test Split** | Holdout, stratified, cross-validation |
| [6](#-tahap-6-pemilihan--pelatihan-algoritma) | **Pilih Algoritma** | 9 algoritma, kapan pakai apa |
| [7](#-tahap-7-evaluasi-model) | **Evaluasi** | Metrik klasifikasi & regresi, overfitting |
| [8](#-tahap-8-hyperparameter-tuning) | **Hyperparameter Tuning** | Grid search, random search, Optuna |
| [9](#-tahap-9-deployment--monitoring) | **Deployment** | Simpan model, REST API, monitoring |

---

## 🧠 Apa itu Supervised Learning?

**Supervised learning** adalah cabang machine learning di mana model dilatih menggunakan data yang sudah diberi label. Setiap sampel terdiri dari pasangan **input (fitur)** dan **output (target/label)** yang benar.

```
Input (X)          →      Model       →     Output (ŷ)
──────────────────        ─────────         ──────────
[ukuran, lokasi]   →   Price Model   →    Rp 850.000.000
[email text]       →   Spam Filter   →    SPAM / HAM
[foto MRI]         →   Cancer Model  →    Positif / Negatif
```

### Dua Tipe Utama

| Tipe | Output | Contoh |
|------|--------|--------|
| 🔵 **Klasifikasi** | Kategori diskret | Spam/bukan spam, kucing/anjing, positif/negatif |
| 📈 **Regresi** | Nilai numerik kontinu | Harga rumah, suhu besok, harga saham |

---

## 🎯 Tahap 1: Definisi Masalah

> Tahap ini sering diremehkan pemula, padahal merupakan **fondasi segalanya**. Model terbaik pun tidak berguna jika dibangun untuk menjawab pertanyaan yang salah.

### Yang Harus Dilakukan

**1. Identifikasi tujuan bisnis secara konkret**

```
❌  "Meningkatkan penjualan"           (terlalu ambigu)
✅  "Prediksi pelanggan mana yang akan berhenti
     berlangganan dalam 30 hari ke depan"
```

**2. Tentukan tipe masalah ML**

```
Apakah outputnya kategori?    →  Klasifikasi
Apakah outputnya angka?       →  Regresi
Apakah outputnya urutan?      →  Ranking / Sequence modeling
```

**3. Pilih metrik evaluasi SEBELUM mulai**

> ⚠️ Jangan pilih metrik setelah melihat hasil — itu bias.

| Masalah | Metrik yang Tepat | Alasan |
|---------|-------------------|--------|
| Deteksi kanker | **Recall** | False negative jauh lebih berbahaya |
| Spam filter | **Precision** | False positive mengganggu pengguna |
| Prediksi harga | **MAE / RMSE** | Ukur error dalam satuan nyata |
| Klasifikasi seimbang | **F1-Score** | Seimbang precision & recall |

### ✅ Checklist Tahap 1

- [ ] Problem statement tertulis jelas (satu paragraf)
- [ ] Tipe task ditentukan: klasifikasi / regresi
- [ ] Metrik sukses dipilih dan dipahami stakeholder
- [ ] Baseline performance ditetapkan
- [ ] Ketersediaan data berlabel dikonfirmasi

---

## 🔍 Tahap 2: Pengumpulan Data & EDA

> *"Kualitas data menentukan kualitas model. Tidak ada algoritma yang bisa mengkompensasi data yang buruk."*
> **— Garbage In, Garbage Out**

### Pemeriksaan Struktural Awal

```python
# Wajib dijalankan pertama kali
print(df.shape)           # jumlah baris × kolom
print(df.dtypes)          # tipe data tiap kolom
print(df.isnull().sum())  # missing values per kolom
print(df.describe())      # statistik deskriptif
print(df.duplicated().sum())  # duplikat
```

### Checklist EDA

```
📊 Distribusi
   ├── Histogram untuk fitur numerik
   ├── Bar chart untuk fitur kategorikal
   └── Boxplot untuk deteksi outlier (IQR / Z-score)

🎯 Target Variable
   ├── Distribusi kelas (klasifikasi) — seimbang?
   └── Distribusi nilai (regresi) — normal? skewed?

🔗 Relasi
   ├── Heatmap korelasi antar fitur
   ├── Scatter plot fitur vs target
   └── Pairplot untuk multi-dimensi

⚠️ Masalah Data
   ├── Missing values > 5%? → perlu strategi imputation
   ├── Kelas tidak seimbang (< 20:80)? → perlu SMOTE / class_weight
   └── Outlier ekstrem? → cap atau transform
```

---

## 🧹 Tahap 3: Data Preprocessing

### Menangani Missing Values

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Strategi berdasarkan situasi:
```

| Strategi | Kapan Digunakan | Kode |
|----------|----------------|------|
| **Drop** | Missing < 5%, pola acak (MCAR) | `df.dropna()` |
| **Mean imputation** | Numerik, distribusi normal | `SimpleImputer(strategy='mean')` |
| **Median imputation** | Numerik, ada outlier | `SimpleImputer(strategy='median')` |
| **Mode imputation** | Kategorikal | `SimpleImputer(strategy='most_frequent')` |
| **KNN imputation** | Data dengan pola, lebih akurat | `KNNImputer(n_neighbors=5)` |
| **MICE / Iterative** | Missing kompleks, paling akurat | `IterativeImputer()` |

### Encoding Variabel Kategorikal

```python
# Label Encoding — untuk ORDINAL (ada urutan)
mapping = {'kecil': 0, 'sedang': 1, 'besar': 2}
df['ukuran_enc'] = df['ukuran'].map(mapping)

# One-Hot Encoding — untuk NOMINAL (tidak ada urutan)
df = pd.get_dummies(df, columns=['kota'], drop_first=True)

# Target Encoding — untuk high-cardinality features
# Encode berdasarkan mean target per kategori
```

### Feature Scaling

> ⚠️ **Aturan Paling Penting:** Fit scaler **HANYA** pada training data!

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # ✅ fit + transform
X_test_scaled  = scaler.transform(X_test)        # ✅ transform saja
# X_test_scaled = scaler.fit_transform(X_test)   # ❌ DATA LEAKAGE!
```

| Scaler | Hasil | Gunakan Jika |
|--------|-------|--------------|
| `StandardScaler` | mean=0, std=1 | Distribusi mendekati normal |
| `MinMaxScaler` | Rentang [0, 1] | Neural network, distribusi non-normal |
| `RobustScaler` | Pakai median & IQR | Data penuh outlier |

**Perlu scaling:** SVM, KNN, MLP, Logistic Regression, PCA
**Tidak perlu scaling:** Decision Tree, Random Forest, XGBoost, LightGBM

---

## ⚙️ Tahap 4: Feature Engineering

> *Feature engineering yang baik lebih berdampak daripada memilih algoritma yang lebih kompleks. Seorang data scientist berpengalaman menghabiskan 60-80% waktunya di tahap ini.*

### Membuat Fitur Baru

```python
# Rasio & kombinasi matematika
df['bmi']              = df['berat_kg'] / (df['tinggi_m'] ** 2)
df['petal_to_sepal']   = df['petal_length'] / df['sepal_length']

# Ekstraksi dari tanggal/waktu
df['hari_dalam_minggu'] = df['tanggal'].dt.dayofweek
df['bulan']             = df['tanggal'].dt.month
df['is_weekend']        = df['hari_dalam_minggu'].isin([5, 6]).astype(int)

# Agregasi per grup
df['avg_purchase_per_user'] = df.groupby('user_id')['amount'].transform('mean')

# Transformasi untuk distribusi skewed
df['log_income'] = np.log1p(df['income'])  # log(x + 1), aman untuk x=0
```

### Feature Selection

```python
# Filter Method — F-test
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)

# Embedded Method — Feature Importance dari Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=feature_names)
importances.sort_values(ascending=False).head(10)

# Wrapper Method — Recursive Feature Elimination
from sklearn.feature_selection import RFE
rfe = RFE(estimator=rf, n_features_to_select=10)
X_rfe = rfe.fit_transform(X_train, y_train)
```

---

## ✂️ Tahap 5: Pembagian Data

> ⚠️ **Aturan emas:** Test set adalah data yang hanya boleh dilihat **sekali** — di evaluasi akhir.

### Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% untuk test
    random_state=42,  # reproducible
    stratify=y        # ✅ WAJIB untuk klasifikasi
)
```

### Strategi Pembagian

| Strategi | Kapan | Proporsi |
|----------|-------|----------|
| **Holdout** | Dataset besar (> 10.000 baris) | 80 / 20 |
| **Train/Val/Test** | Saat hyperparameter tuning | 60 / 20 / 20 |
| **Stratified** | Klasifikasi, kelas tidak seimbang | sesuaikan |
| **Time-based** | Data sekuensial (time series) | data lama → train, baru → test |

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Stratified K-Fold — gunakan ini untuk klasifikasi
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')

print(f"CV Score: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 🤖 Tahap 6: Pemilihan & Pelatihan Algoritma

> *Tidak ada algoritma terbaik untuk semua masalah — "No Free Lunch Theorem"*

### Panduan Memilih Algoritma

```
Dataset kecil (< 1000 sampel)?
├── Perlu interpretabilitas tinggi?  →  Logistic / Linear Regression, Decision Tree
├── Akurasi prioritas?               →  SVM, Random Forest
└── Data high-dimensional?           →  SVM, Naive Bayes

Dataset medium (1K – 100K sampel)?
├── Baseline cepat?                  →  Logistic / Linear Regression
├── Default terbaik (tabular)?       →  Random Forest ⭐
└── Performa maksimal (tabular)?     →  XGBoost / LightGBM ⭐

Dataset besar (> 100K sampel)?
├── Perlu interpretabilitas?         →  Logistic Regression + SGD
└── Performa maksimal?               →  LightGBM / Neural Network
```

### Perbandingan 9 Algoritma

| Algoritma | Tipe | Kelebihan | Kekurangan | Cocok untuk |
|-----------|------|-----------|------------|-------------|
| **Logistic Regression** | Klasifikasi | Interpretable, cepat, baseline baik | Hanya linear | Dataset kecil-medium, NLP |
| **Decision Tree** | Keduanya | Mudah dijelaskan, no scaling | Sangat overfit | Prototyping, presentasi bisnis |
| **Random Forest** | Keduanya | Robust, feature importance, akurat | Lambat, less interpretable | Data tabular umum ⭐ |
| **Gradient Boosting** | Keduanya | Akurasi tinggi, fleksibel | Lambat, banyak hyperparameter | Kompetisi, performa maks |
| **XGBoost / LightGBM** | Keduanya | Terbaik untuk tabular, paralel | Banyak hyperparameter | Kaggle, production tabular ⭐ |
| **SVM** | Klasifikasi | Efektif high-dim, robust | Lambat data besar | Teks, bioinformatika |
| **KNN** | Keduanya | Sederhana, no training | Lambat inferensi, sensitif skala | Dataset kecil, recommendation |
| **Naive Bayes** | Klasifikasi | Sangat cepat, bagus NLP | Asumsi independensi kuat | Spam detection, klasifikasi teks |
| **MLP / Neural Net** | Keduanya | Sangat fleksibel, data kompleks | Butuh banyak data & tuning | Gambar, suara, tabular besar |

### Training dengan Pipeline (Best Practice)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ✅ Pipeline memastikan tidak ada data leakage
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

---

## 📊 Tahap 7: Evaluasi Model

### Metrik Klasifikasi

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Laporan lengkap per kelas
print(classification_report(y_test, y_pred, target_names=class_names))

# ROC-AUC (multi-class)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
```

| Metrik | Formula | Gunakan Jika |
|--------|---------|--------------|
| **Accuracy** | (TP+TN) / Total | Kelas seimbang |
| **Precision** | TP / (TP+FP) | False positive mahal (spam filter) |
| **Recall** | TP / (TP+FN) | False negative berbahaya (deteksi kanker) |
| **F1-Score** | 2×P×R / (P+R) | Perlu keseimbangan precision & recall |
| **ROC-AUC** | Area under ROC curve | Evaluasi threshold-independent |
| **PR-AUC** | Area under PR curve | Kelas sangat tidak seimbang |

### Metrik Regresi

```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R²   : {r2:.4f}")   # 1.0 = sempurna
print(f"MAE  : {mae:.2f}")  # dalam satuan target
print(f"RMSE : {rmse:.2f}") # lebih sensitif terhadap error besar
```

### Diagnosis Overfitting vs Underfitting

```
Train Score   Test Score   Diagnosis        Solusi
──────────    ──────────   ─────────────    ────────────────────────────────
Rendah        Rendah       Underfitting     Tambah kompleksitas, fitur baru,
                                            kurangi regularisasi
Tinggi        Rendah       Overfitting      Regularisasi, more data, dropout,
                                            pruning, early stopping
Tinggi        Tinggi       ✅ Good Fit      Lanjutkan ke tuning & deployment
```

---

## 🎛️ Tahap 8: Hyperparameter Tuning

> Hyperparameter adalah setelan yang kita tentukan **sebelum** training — tidak dipelajari dari data.

### Metode Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# GridSearchCV — exhaustive, cocok untuk ruang kecil
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 5, 10],
    'model__min_samples_split': [2, 5],
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# RandomizedSearchCV — sampling acak, lebih cepat ⭐
from scipy.stats import randint
param_dist = {
    'model__n_estimators': randint(50, 500),
    'model__max_depth': [None, 3, 5, 7, 10],
}
random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=50, cv=5, n_jobs=-1)
random_search.fit(X_train, y_train)

print(f"Best params : {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
```

### Hyperparameter Penting per Algoritma

<details>
<summary><b>🌲 Random Forest</b></summary>

```python
{
    'n_estimators': [100, 200, 500],       # lebih banyak = lebih stabil
    'max_depth': [None, 5, 10, 20],        # None = pohon penuh
    'min_samples_split': [2, 5, 10],       # min sampel untuk split node
    'min_samples_leaf': [1, 2, 4],         # min sampel di daun
    'max_features': ['sqrt', 'log2', None] # fitur per split
}
```
</details>

<details>
<summary><b>🚀 XGBoost / LightGBM</b></summary>

```python
{
    'learning_rate': [0.01, 0.05, 0.1, 0.3],  # lebih kecil = lebih akurat, lambat
    'n_estimators': [100, 300, 500, 1000],
    'max_depth': [3, 4, 5, 6],                 # 3-6 optimal untuk boosting
    'subsample': [0.7, 0.8, 1.0],              # row sampling per tree
    'colsample_bytree': [0.7, 0.8, 1.0]        # feature sampling per tree
}
```
</details>

<details>
<summary><b>🧲 SVM</b></summary>

```python
{
    'C': [0.1, 1, 10, 100],          # regularisasi: kecil=smooth, besar=fit ketat
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto', 0.1]  # lebar kernel RBF
}
```
</details>

<details>
<summary><b>🧠 MLP / Neural Network</b></summary>

```python
{
    'hidden_layer_sizes': [(100,), (100,50), (128,64,32)],
    'learning_rate_init': [0.001, 0.01],
    'alpha': [0.0001, 0.001, 0.01],  # L2 regularization
    'activation': ['relu', 'tanh']
}
```
</details>

---

## 🚀 Tahap 9: Deployment & Monitoring

### Menyimpan & Memuat Model

```python
import joblib

# Simpan
joblib.dump(pipeline, 'model_v1.pkl')
print("✅ Model tersimpan!")

# Load & prediksi
model = joblib.load('model_v1.pkl')
prediksi = model.predict(data_baru)
probabilitas = model.predict_proba(data_baru)
```

### REST API dengan FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np

app = FastAPI(title="ML Model API")
model = joblib.load('model_v1.pkl')

class InputData(BaseModel):
    fitur1: float
    fitur2: float
    fitur3: float

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.fitur1, data.fitur2, data.fitur3]])
    prediksi = model.predict(X)[0]
    proba    = model.predict_proba(X)[0].max()
    return {"prediksi": int(prediksi), "confidence": round(float(proba), 4)}

# Jalankan: uvicorn app:app --reload
```

### Opsi Deployment

| Metode | Cocok untuk | Tools |
|--------|-------------|-------|
| **REST API** | Prediksi real-time, integrasi sistem | FastAPI, Flask |
| **Batch prediction** | Prediksi jutaan record, scheduled | Airflow, cron job |
| **Embedded** | Mobile, edge device | ONNX, TFLite |
| **Cloud ML** | Skalabilitas tinggi, managed | AWS SageMaker, GCP Vertex |

### Monitoring Model di Production

```
📊 Data Drift      → Distribusi input berubah dari waktu training?
                     Gunakan: Evidently AI, Alibi Detect, NannyML

📉 Model Drift     → Performa model menurun seiring waktu?
                     Pantau: metrik evaluasi pada data terbaru secara berkala

🔄 Retraining      → Kapan harus retrain?
                     Trigger: performa turun > threshold, data drift terdeteksi,
                     setiap X hari/bulan (scheduled retraining)
```

---

## ⚠️ Kesalahan Umum yang Harus Dihindari

```
❌  Data Leakage
    Fit scaler/imputer pada seluruh dataset (termasuk test set)
    → Gunakan Pipeline, fit HANYA pada training data

❌  Melihat test set berkali-kali
    Test set digunakan untuk memilih model/hyperparameter
    → Gunakan validation set atau cross-validation untuk tuning

❌  Mengabaikan class imbalance
    Accuracy 95% terlihat bagus, tapi kelas minoritas recall = 0%
    → Cek confusion matrix, gunakan F1/ROC-AUC, pertimbangkan SMOTE

❌  Feature engineering setelah split
    Membuat fitur baru menggunakan statistik dari seluruh dataset
    → Selalu split dulu, baru feature engineering pada train set

❌  Langsung pakai model kompleks
    Langsung XGBoost / Neural Network tanpa baseline
    → Mulai dari Logistic/Linear Regression, build up complexity

❌  Tidak mencatat eksperimen
    Lupa hyperparameter apa yang dipakai, hasil mana yang terbaik
    → Gunakan MLflow, Weights & Biases, atau minimal spreadsheet
```

---

## 🏃 Quick Start

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm joblib
```

### Menjalankan Notebook di Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/username/repo/blob/main/Supervised_Learning_Lengkap.ipynb)

```bash
# Clone repository
git clone https://github.com/username/supervised-learning-guide.git
cd supervised-learning-guide

# Install dependencies
pip install -r requirements.txt

# Jalankan notebook
jupyter notebook Supervised_Learning_Lengkap.ipynb
```

### Struktur Proyek

```
supervised-learning-guide/
├── 📓 Supervised_Learning_Lengkap.ipynb   # Notebook utama (9 tahap lengkap)
├── 🐍 ml_comparison.py                    # Script perbandingan 9 algoritma
├── 📊 outputs/
│   ├── clf_iris.png                       # Plot hasil klasifikasi Iris
│   ├── clf_breast_cancer.png              # Plot hasil klasifikasi Breast Cancer
│   └── reg_diabetes.png                  # Plot hasil regresi Diabetes
└── 📖 README.md                           # Panduan ini
```

---

## 📈 Hasil Eksperimen

### Dataset Iris (Klasifikasi, 150 sampel)

| Model | CV Accuracy | Test Accuracy |
|-------|-------------|---------------|
| Random Forest | **96.7%** | **96.7%** |
| SVM | **96.7%** | 96.7% |
| Gradient Boosting | **96.7%** | 96.7% |
| Logistic Regression | 95.8% | 96.7% |
| KNN | 95.8% | 96.7% |

### Dataset Breast Cancer (Klasifikasi, 569 sampel)

| Model | CV Accuracy | Test Accuracy |
|-------|-------------|---------------|
| **Logistic Regression** | **98.1%** | **97.4%** |
| Random Forest | 96.2% | 96.5% |
| SVM | 97.4% | 97.4% |

> 💡 *Menarik: Model sederhana (Logistic Regression) mengalahkan model kompleks di dataset ini!*

### Dataset Diabetes (Regresi, 442 sampel)

| Model | CV R² | RMSE |
|-------|-------|------|
| **Lasso** | **0.4825** | 54.2 |
| Gradient Boosting | 0.4710 | 55.1 |
| Random Forest | 0.4580 | 56.2 |

---

## 📚 Referensi & Bacaan Lanjutan

| Topik | Sumber |
|-------|--------|
| Scikit-learn documentation | [scikit-learn.org](https://scikit-learn.org/stable/) |
| XGBoost | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) |
| Feature Engineering | *Feature Engineering for Machine Learning* — Alice Zheng |
| ML Best Practices | *Hands-On ML* — Aurélien Géron |
| Interpretable ML | [christophm.github.io/interpretable-ml-book](https://christophm.github.io/interpretable-ml-book/) |
| MLOps / Deployment | [ml-ops.org](https://ml-ops.org/) |

---

<div align="center">

**Dibuat dengan ❤️ untuk komunitas data science Indonesia**

*Jika panduan ini bermanfaat, jangan lupa ⭐ repo ini!*

<br/>

[![GitHub stars](https://img.shields.io/github/stars/username/repo?style=social)](https://github.com/username/repo)
[![GitHub forks](https://img.shields.io/github/forks/username/repo?style=social)](https://github.com/username/repo)

</div>
