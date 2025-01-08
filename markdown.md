# Laporan Proyek Machine Learning - I Gusti Bagus Ramadha Saverian Ranuh

## Domain Proyek


### Latar Belakang:

Kesehatan maternal merupakan aspek penting dalam sistem kesehatan global, karena berkaitan langsung dengan kesejahteraan ibu dan bayi yang dikandungnya. Berdasarkan laporan dari World Health Organization (WHO), komplikasi yang terjadi selama kehamilan dan persalinan menjadi salah satu penyebab utama kematian wanita usia reproduksi, terutama di negara-negara berkembang. Masalah ini tidak hanya memengaruhi tingkat kelangsungan hidup ibu, tetapi juga meningkatkan risiko kematian neonatal.

### **Rubrik/Kriteria Tambahan**:

Komplikasi seperti preeklamsia, hipertensi gestasional, dan diabetes gestasional adalah beberapa kondisi yang sering terjadi selama kehamilan dan berpotensi membahayakan. Penelitian yang dipublikasikan di jurnal The Lancet menyebutkan bahwa hipertensi selama kehamilan menjadi penyebab utama preeklamsia, yang memengaruhi sekitar 2-8% kehamilan secara global. Sementara itu, kadar glukosa darah yang tinggi selama kehamilan meningkatkan risiko persalinan prematur dan komplikasi lain yang dapat mengancam jiwa.

Pemantauan indikator kesehatan seperti tekanan darah, kadar glukosa darah, detak jantung, dan usia ibu merupakan langkah penting untuk mendeteksi dini risiko kesehatan maternal. Di sinilah teknologi Internet of Things (IoT) memberikan solusi inovatif. Melalui sistem pemantauan berbasis IoT, data dapat dikumpulkan secara real-time dari rumah sakit, klinik komunitas, atau layanan kesehatan maternal. Data ini, jika dianalisis menggunakan teknik pembelajaran mesin, dapat memberikan prediksi akurat mengenai tingkat risiko yang dihadapi ibu hamil.

Proyek "Maternal Health Risk" bertujuan untuk memanfaatkan data yang dikumpulkan melalui IoT untuk memprediksi intensitas risiko selama kehamilan. Dengan demikian, proyek ini menawarkan pendekatan berbasis data untuk mengurangi angka kematian maternal dan neonatal dengan menyediakan peringatan dini kepada penyedia layanan kesehatan dan ibu hamil.

  Referensi:

  - [Trends in maternal mortality 2000 to 2017](https://www.unfpa.org/sites/default/files/pub-pdf/Maternal_mortality_report.pdf)
  - [Hypertensive disorders in pregnancy: prevalence, risk factors, and outcomes](https://doi.org/10.1016/j.preghy.2021.02.005)
  - [Gestational Diabetes Mellitus: Risk Factors and Management Approaches](10.1038/nrendo.2012.96.)

## Business Understanding

### Problem Statement

- Tingginya angka kematian ibu hamil akibat komplikasi kehamilan sering kali disebabkan oleh kurangnya deteksi dini terhadap risiko kesehatan maternal.
- Data kesehatan seperti tekanan darah, kadar glukosa, dan detak jantung sering kali belum dimanfaatkan secara optimal untuk membuat prediksi yang akurat mengenai risiko kesehatan ibu.
- Kurangnya sistem pemantauan berbasis data yang andal, terutama di negara berkembang, menyebabkan keterlambatan dalam pengambilan keputusan medis.

### Goals 

- Membangun model prediksi yang akurat untuk menentukan tingkat risiko kesehatan maternal berdasarkan data kesehatan (usia, tekanan darah, kadar glukosa, detak jantung).
- Menggunakan data dari sistem pemantauan berbasis IoT untuk mendukung decision making medis secara lebih cepat dan akurat.
- Membandingkan performa beberapa algoritma machine learning untuk menentukan pendekatan terbaik dalam memprediksi risiko maternal.

### **Rubrik/Kriteria Tambahan**:
### Solution Statements
Untuk mencapai tujuan di atas, berikut adalah solusi yang diajukan:
#### 1. Eksperimen dengan Beberapa Algoritma

Beberapa algoritma pembelajaran mesin akan diterapkan untuk memprediksi risiko kesehatan maternal:
- K-Nearest Neighbors (KNN): Algoritma berbasis tetangga terdekat yang sederhana namun efektif untuk masalah klasifikasi.
- Support Vector Machine (SVM): Algoritma yang mampu memisahkan kelas dengan margin optimal, cocok untuk data dengan fitur kompleks.
- Random Forest: Algoritma berbasis ensemble yang dapat menangani data dengan non-linearitas dan fitur yang saling berinteraksi.

#### 2. Peningkatan Model dengan Hyperparameter Tuning
- Menggunakan GridSearchCV untuk mencari kombinasi parameter terbaik bagi setiap algoritma.
- Contoh parameter yang akan disesuaikan:
- Untuk KNN: jumlah tetangga (n_neighbors).
- Untuk SVM: kernel (linear, rbf) dan parameter regulasi (C).
- Untuk Random Forest: jumlah pohon (n_estimators) dan kedalaman maksimum pohon (max_depth).

#### 3. Evaluasi dengan Metrik yang Terukur. Metrik evaluasi yang digunakan:
- Accuracy: Mengukur persentase prediksi yang benar.
- Precision: Mengukur akurasi prediksi kelas positif.
- Recall: Mengukur sensitivitas atau kemampuan model mendeteksi kelas positif.
- F1-score: Rata-rata harmonik antara precision dan recall.
- AUC-ROC: Mengukur kemampuan model dalam membedakan kelas.

#### 4. Analisis dan Perbandingan Hasil
- Membandingkan hasil dari masing-masing algoritma berdasarkan metrik evaluasi.
- Memilih algoritma dengan performa terbaik untuk implementasi lebih lanjut.


## Data Understanding


### **1. Informasi tentang Data**

#### **Jumlah Data**
Dataset terdiri dari **1.014 baris (observasi data IoT)** dan **6 kolom (feature)**, termasuk label target, yaitu *Risk Level*.  

#### **Kondisi Data**
- **Tidak ada data kosong (missing values):** Dataset bersih dari nilai yang hilang.  
- **Tidak ada data duplikat:** Dataset tidak memiliki baris yang sama persis.  
- **Variabel target:** *Risk Level* memiliki tiga kategori (Low, Mid, High).  

#### **Sumber Data**
[Maternal Health Risk Data](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data/data)  

#### **Uraian Seluruh Variabel**
| **Nama Fitur** | **Deskripsi**                                               | **Tipe Data** |
|-----------------|-----------------------------------------------------------|---------------|
| **Age**         | Usia ibu hamil dalam tahun.                                | Numerik       |
| **SystolicBP**  | Tekanan darah atas dalam mmHg.                             | Numerik       |
| **DiastolicBP** | Tekanan darah bawah dalam mmHg.                            | Numerik       |
| **BS**          | Kadar glukosa darah dalam mmol/L.                          | Numerik       |
| **HeartRate**   | Detak jantung dalam *beats per minute* (BPM).              | Numerik       |
| **Risk Level**  | Tingkat risiko kesehatan maternal (Low, Mid, High).        | Kategorikal   |

---

### **2. Exploratory Data Analysis (EDA)**

#### **Statistik Deskriptif**
| **Fitur**       | **Mean** | **Median** | **Std Dev** |
|------------------|----------|------------|-------------|
| **Age**          | 29.7     | 30         | 6.5         |
| **SystolicBP**   | 119.1    | 118        | 12.8        |
| **DiastolicBP**  | 76.2     | 76         | 8.6         |
| **BS**           | 6.5      | 6.3        | 1.8         |
| **HeartRate**    | 76.3     | 76         | 6.5         |

#### **Distribusi Variabel Numerik**
- **Histogram:** 
  - *Age:* Mayoritas usia ibu hamil berada di rentang 25–35 tahun.  
  - *SystolicBP:* Sebagian besar tekanan darah sistolik berkisar antara 110–130 mmHg.  
  - *HeartRate:* Detak jantung cenderung normal di sekitar 70–80 BPM.

#### **Distribusi Variabel Kategorikal**
- **Bar Plot untuk Risk Level:**  
  - **Low:** 406 (40%)  
  - **Mid:** 336 (33%)  
  - **High:** 272 (27%)  

#### **Hubungan antar Variabel**
- Korelasi Pearson untuk variabel numerik:
  - Tekanan darah sistolik dan diastolik memiliki korelasi positif tinggi (r = 0.79).
  - Kadar glukosa darah tidak berkorelasi signifikan dengan usia atau detak jantung.
- Visualisasi **Heatmap** untuk melihat korelasi antar fitur.

#### **Analisis Risiko Berdasarkan Fitur**
- **Box Plot**: Membandingkan distribusi fitur untuk setiap kategori *Risk Level*.  
  - **High Risk:** Lebih sering terjadi pada kadar glukosa darah (*BS*) > 8 mmol/L dan tekanan darah sistolik > 120 mmHg.  
  - **Low Risk:** Umumnya terkait dengan tekanan darah normal dan kadar glukosa rendah.  

#### **Outlier Detection**
- **Box Plot** menunjukkan beberapa outlier:  
  - Tekanan darah sistolik > 180 mmHg.  
  - Nilai kadar glukosa darah melebihi 12 mmol/L.

#### **Distribusi Multivariat**
- **Scatter Plot** untuk hubungan antara *SystolicBP* dan *DiastolicBP* berdasarkan *Risk Level*.  
  - **High Risk** terlihat lebih sering terjadi pada pasangan tekanan darah yang tinggi.

---

### **Kesimpulan Sementara dari EDA**
- Risiko kesehatan maternal memiliki hubungan yang kuat dengan tekanan darah dan kadar glukosa darah.  
- Visualisasi membantu memahami pola risiko berdasarkan variabel, yang akan memandu pemilihan model pembelajaran mesin.


## Data Preparation

### **1. Teknik Data Preparation yang Dilakukan**
Data preparation adalah tahap penting sebelum pemodelan karena memastikan data bersih, relevan, dan siap untuk diterapkan pada algoritma pembelajaran mesin. Berdasarkan referensi yang diberikan, berikut teknik-teknik yang dilakukan:

#### **Tahapan Data Preparation**
1. **Import Library dan Dataset**
2. **Pemeriksaan Data Awal**
   - Cek informasi dataset (*missing values*, duplikasi, dan tipe data).
   - Deskripsi statistik dasar.
3. **Transformasi dan Pembersihan Data**
   - Encode variabel kategorikal menggunakan *Label Encoding*.
4. **Normalisasi Data**
   - Normalisasi variabel numerik menggunakan *MinMaxScaler*.
5. **Pemisahan Data (Train-Test Split)**
   - Membagi dataset menjadi set pelatihan (80%) dan set pengujian (20%).

---

### **2. Penjelasan Proses Data Preparation**

#### **Step 1: Import Library dan Dataset**
- **Tujuan:** Mengimpor pustaka seperti `pandas`, `numpy`, `sklearn`, dan `matplotlib` untuk analisis data dan visualisasi.
- **Kode:**
  ```python
  import pandas as pd
  import numpy as np
  from sklearn.preprocessing import LabelEncoder, StandardScaler
  from sklearn.model_selection import train_test_split
  ```


#### **Step 2: Pemeriksaan Data Awal**
- **Teknik:** Menggunakan fungsi seperti `.info()`, `.describe()`, `.isnull().sum()`, dan `.duplicated().sum()` untuk menganalisis struktur data.
- **Alasan:** 
  - Memastikan tidak ada nilai yang hilang (*missing values*), duplikasi, atau inkonsistensi.
  - Memberikan gambaran awal mengenai distribusi data dan tipe data.
- **Kode:**
  ```python
  data.info()
  data.describe()
  data.isnull().sum()
  data.duplicated().sum()
  ```

#### **Step 3: Transformasi dan Pembersihan Data**
- **Teknik:** Menggunakan Label Encoding untuk mengubah target variable `Risk Level` dari kategorikal menjadi numerik.

- Hal ini dilakukan agar model machine learning hanya menerima output numerik. Transformasi ini membantu model memahami hubungan konteks antara variable kategori.

- **Kode:**
  ```python
  data.replace({"high risk": 2, "low risk": 0, "mid risk": 1}, inplace=True)
  ```

#### **Step 4: Train Test Split**

- **Teknik:** Membagi dataset menjadi 80% data train dan 20% data test

- **Alasan:** Memisahkan train dataset dan test dataset untuk memastikan evaluasi model dilakukan pada data yang belum pernah dilihat model.

- **Kode:**
  ```python
  X = data.drop('RiskLevel', axis=1)
  y = data['RiskLevel']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
  ```

#### **Step 5: Normalisasi Data**

- **Teknik:** Menggunakan *MinMaxScaler* untuk menormalkan variabel numerik (`Age`, `SystolicBP`, `DiastolicBP`, `BS`, `HeartRate`)

- **Alasan:** Normalisasi ini membantu mengurangi skala feature yang berbeda agar model dapat belajar secara optimal, khususnya algoritma berbasis jarak seperti KNN.

- **Kode:**
  ```python
  scaler = StandardScaler()
  scaler.fit(X) # Fit data
  X_scaled = scaler.transform(X)
  ```

### 3. Aalsan Diperlukan Tahapan Data Preparation

1. **Pemeriksaan Data Awal:** Menjamin data dalam kondisi baik sebelum digunakan dalam pemodelan.
2. **Transformasi Data:** Variabel target perlu diubah menjadi format yang dapat diterima oleh model.
3. **Normalisasi:** Mengatasi perbedaan skala fitur agar model lebih stabil dan akurat.
4. **Train-Test Split:** Memisahkan data untuk evaluasi yang tidak bias.

## Modeling

### **Tahapan Pemodelan**

#### **1. K-Nearest Neighbors (KNN)**
- **Parameter Awal:**
  - `n_neighbors=1`
- **Hasil Awal:**
  - Akurasi: 55%
  - Precision, Recall, dan F1-score yang rendah pada kategori `Risk Level` 1 dan 2.
- **Improvement:**
  - Meningkatkan nilai `n_neighbors` menjadi 16 untuk mengurangi *overfitting* dan menangkap pola lebih baik.
- **Hasil Setelah Improvement:**
  - Akurasi meningkat menjadi 66%.
  - Kinerja kategori `Risk Level` 0 dan 2 membaik, namun kategori 1 masih memiliki kinerja rendah.
- **Kelebihan:**
  - Sederhana dan mudah diimplementasikan.
  - Tidak memerlukan asumsi distribusi data.
- **Kekurangan:**
  - Sensitif terhadap skala fitur, sehingga normalisasi diperlukan.
  - Performa menurun dengan jumlah data besar atau tinggi dimensi.

---

#### **2. Support Vector Classifier (SVC)**
- **Parameter:**
  - Menggunakan *class_weight* untuk menangani ketidakseimbangan kelas.
- **Hasil:**
  - Akurasi: 67%.
  - Precision dan Recall kategori `Risk Level` 0 dan 2 cukup baik, namun kategori 1 masih rendah.
- **Kelebihan:**
  - Efektif untuk data dengan margin yang jelas antar kelas.
  - Mendukung kernel untuk menangani data non-linear.
- **Kekurangan:**
  - Kurang optimal untuk dataset besar.
  - Memerlukan tuning parameter seperti `C` dan kernel untuk performa optimal.

---

#### **3. Random Forest**
- **Parameter Awal:**
  - *GridSearchCV* digunakan untuk mencari parameter terbaik:
    - `n_estimators`: [100, 300, 500]
    - `criterion`: ['gini', 'entropy']
    - `max_depth`: [20, 25, 30]
    - `min_samples_leaf`: [2, 3, 5]
- **Hasil Grid Search:**
  - Parameter terbaik: `{'criterion': 'entropy', 'max_depth': 30, 'min_samples_leaf': 5, 'n_estimators': 500}`
  - Akurasi terbaik: 70.6%.
- **Hasil Akhir:**
  - Akurasi: 71%.
  - Precision dan Recall kategori `Risk Level` 0 dan 2 sangat baik, kategori 1 masih memiliki kinerja moderat.
- **Kelebihan:**
  - Mengatasi *overfitting* dengan *bagging*.
  - Memberikan fitur penting dalam prediksi.
- **Kekurangan:**
  - Kurang interpretatif dibandingkan model sederhana.
  - Memerlukan lebih banyak sumber daya komputasi.

---

### **Pemilihan Model Terbaik**
- Model **Random Forest** dipilih sebagai model terbaik:
  - Akurasi tertinggi (71%) dibandingkan KNN dan SVC.
  - Kinerja yang lebih konsisten pada kategori `Risk Level` 0, 1, dan 2.
  - Dapat menangani ketidakseimbangan kelas dengan baik.

---

### **Alasan Dilakukannya Improvement**
- **KNN:** Hyperparameter tuning pada `n_neighbors` untuk menangkap pola data lebih baik.
- **SVC:** Penyesuaian *class_weight* untuk menangani ketidakseimbangan kelas.
- **Random Forest:** *GridSearchCV* digunakan untuk mencari parameter optimal, karena model ini sensitif terhadap pengaturan parameter seperti jumlah pohon (*n_estimators*) dan kedalaman maksimal (*max_depth*).

---

### **Kesimpulan**
- Random Forest memberikan hasil terbaik dalam hal akurasi dan konsistensi antar kategori risiko.
- Proses improvement membantu setiap model meningkatkan kinerjanya dengan meminimalkan *bias* atau *variance*.

## Evaluation

### **Metrik Evaluasi yang Digunakan**
1. **Confusion Matrix**  
   - *Confusion Matrix* memberikan informasi tentang jumlah prediksi yang benar (True Positives dan True Negatives) serta kesalahan (False Positives dan False Negatives) untuk setiap kelas.
   - Matriks ini digunakan untuk menghitung metrik lain seperti Precision, Recall, dan F1-Score.

2. **Precision**  
   - Precision mengukur proporsi prediksi positif yang benar-benar positif. Cocok untuk kasus di mana kesalahan False Positive harus diminimalkan.

3. **Recall (Sensitivity)**  
   - Recall mengukur kemampuan model untuk menangkap semua data positif yang sebenarnya. Cocok untuk kasus di mana kesalahan False Negative perlu diminimalkan.

4. **F1-Score**  
   - Kombinasi Precision dan Recall yang seimbang. Berguna jika ada trade-off antara False Positives dan False Negatives.

5. **Accuracy**  
   - Mengukur persentase prediksi yang benar dari total prediksi. Namun, metrik ini kurang efektif untuk dataset yang tidak seimbang.
  
6. **ROC Curve (Receiver Operating Characteristic)**

- Grafik ROC menunjukkan hubungan antara True Positive Rate (Sensitivity/Recall) dan False Positive Rate (1 - Specificity) pada berbagai threshold keputusan.
- Kurva ini membantu mengevaluasi kemampuan model untuk membedakan antara kelas positif dan negatif secara keseluruhan.

7. **AUC (Area Under the Curve)**
- AUC adalah nilai numerik yang menunjukkan luas di bawah kurva ROC. Semakin tinggi nilai AUC, semakin baik model dalam memisahkan kelas.
- Rentang nilai AUC adalah antara 0.5 (tidak ada kemampuan diskriminasi, setara dengan tebak-tebakan) hingga 1.0 (diskriminasi sempurna).

---

### **Hasil Proyek Berdasarkan Metrik Evaluasi**
1. **K-Nearest Neighbors (KNN)**
   - Akurasi meningkat dari **55% (K=1)** menjadi **66% (K=16)** setelah hyperparameter tuning.
   - Precision dan Recall pada kategori `Risk Level` 0 dan 2 cukup baik setelah improvement, tetapi kategori 1 masih memiliki kinerja rendah karena data yang tidak seimbang.
   - Model KNN memiliki performa yang cukup baik dengan AUC sebesar 0.77. Artinya, pada 77% kasus, model ini dapat membedakan antara kelas positif dan negatif dengan benar.

2. **Support Vector Classifier (SVC)**
   - Akurasi: **67%**.
   - Precision, Recall, dan F1-Score pada kategori `Risk Level` 0 dan 2 memadai, tetapi kategori 1 tetap rendah, menunjukkan kelemahan dalam menangani kelas minoritas meskipun telah menggunakan *class_weight
   - Model SVC memiliki AUC yang sama dengan KNN, yaitu 0.77. Ini menunjukkan bahwa SVC dan KNN memiliki kemampuan diskriminasi yang serupa dalam memisahkan kelas.

3. **Random Forest**
   - Akurasi tertinggi: **73%**.
   - Precision, Recall, dan F1-Score menunjukkan kinerja konsisten pada semua kategori, dengan perbaikan signifikan pada kategori `Risk Level` 1 dibandingkan dengan model lain.
   - Model Random Forest memiliki AUC tertinggi (0.81), menunjukkan performa terbaik dibandingkan dengan KNN dan SVC. Random Forest lebih andal dalam membedakan antara kelas positif dan negatif, menjadikannya model yang lebih cocok untuk masalah ini.

---

### **Penjelasan Metrik Evaluasi**

1. **Kenapa Metrik Ini Digunakan?**
   - **Precision:** Penting untuk memastikan bahwa prediksi positif yang dibuat benar-benar relevan (mengurangi risiko False Positives). 
   - **Recall:** Relevan untuk masalah kesehatan seperti ini, di mana kesalahan dalam memprediksi risiko (False Negatives) bisa berakibat serius.
   - **F1-Score:** Kombinasi Precision dan Recall yang cocok untuk data tidak seimbang seperti `Risk Level`.
   - **Accuracy:** Memberikan gambaran umum tentang kinerja model, tetapi kurang informatif jika data tidak seimbang.

2. **Konteks Data dan Problem Statement**
   - Dataset ini memiliki ketidakseimbangan kelas (`Risk Level` 1 memiliki lebih sedikit data). Oleh karena itu, metrik seperti Precision, Recall, dan F1-Score lebih relevan untuk mengukur kinerja dibandingkan hanya mengandalkan Accuracy.

---

### **Kesimpulan Hasil Evaluasi**
- **Random Forest** adalah model terbaik dengan akurasi tertinggi (**71%**) dan kinerja metrik evaluasi yang paling seimbang untuk semua kategori risiko.
- Penggunaan Precision, Recall, dan F1-Score memberikan wawasan yang lebih baik dalam menilai kemampuan model dalam memprediksi risiko kehamilan yang berbahaya.
- Metrik ini membantu memilih model yang tidak hanya akurat tetapi juga andal dalam menangani kelas minoritas (`Risk Level` 1).
- Random Forest lebih andal dalam membedakan antara kelas positif dan negatif pada metrik evaluasi ROC AUC, menjadikannya model yang lebih cocok untuk masalah ini.


**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
