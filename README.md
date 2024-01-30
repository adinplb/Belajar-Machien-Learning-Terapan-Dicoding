# Laporan Proyek Machine Learning - Muhammad Adin Palimbani

## Domain Proyek
Kemajuan yang signifikan dalam penelitian kanker selama beberapa dekade terakhir telah dilakukan dengan munculnya teknologi baru di bidang kedokteran. Para ilmuwan telah melakukan pendekatan baru dengan metode yang berbeda untuk prediksi awal hasil pengobatan kanker terutama Kanker Payudara. Salah satu contoh pendekatan yang diterapkan adalah tren yang berkembang pada Machine Learning. Namun, [masalah umum dalam beberapa penelitian](https://www.sciencedirect.com/science/article/pii/S1877050921014629) adalah kurangnya validasi eksternal atau pengujian mengenai kinerja prediktif model mereka dan juga menangani data yang tidak seimbang. Ini dapat menyebabkan model prediksi pada kanker yang salah dan kegagalan sistem pada tahap produksi. Model prediksi akurat dari hasil penyakit sangat tergantung pada data medis pasien. Data medis berisi kondisi detail pasien dan diagnosis yang menyimpan data yang tidak perlu dan saling terkait. Dalam beberapa penelitian, para ilmuwan telah membuktikan bahwa pendekatan yang berkaitan dengan karakteristik genom memberikan hasil yang menjanjikan untuk deteksi dan identifikasi kanker, misalnya, gambar digital dari aspirat jarum halus (FNA) pada massa payudara yang mewakili karakteristik sel nukleus dalam Tumor Payudara. Namun, metode ini menderita sensitivitas rendah mengenai penggunaannya dalam skrining pada tahap awal dan kesulitan untuk menentukan jinak dari tumor ganas. Ini adalah alasan mengapa masalah model kinerja prediktif kanker perlu diselesaikan untuk mencegah prediksi yang salah dan kegagalan sistem dalam mendiagnosis apakah itu ganas atau jinak melalui Ekstraksi Fitur Sel Nukleus.

## Business Understanding
Diagnosis Tumor Payudara telah dilakukan dengan metode [Fine Needle Aspiration (FNA)](https://cancer.ca/en/treatments/tests-and-procedures/fine-needle-aspiration-fna); jenis biopsi yang menggunakan jarum dan jarum suntik yang sangat tipis untuk menghilangkan sampel sel, jaringan atau cairan dari area abnormal atau benjolan dalam tubuh. FNA telah berhasil mendiagnosis karakteristik sel nukleus dan menjadi fitur yang menunjukkan kemungkinan keganasan yang lebih tinggi. Sistem diagnostik visi komputer mengekstrak 10 fitur berbeda dari batas inti sel yang dihasilkan ular. Fitur yang diekstraksi dimodelkan secara numerik yang terdiri dari Radius, Perimeter, Area, Kekompakan, Kelancaran, rasa hormat, Poin Cekung, Simetri, Dimensi Fratal dan Tekstur. Untuk menjawab masalah ini, predictive analytics dengan supervised machine learning diharapkan dapat memprediksi masalah tersebut dan mendapatkan solusi yang terbaik dengan menggunakan model machine learning.

### Problem Statement
Berikut adalah problem statement dari proyek ini:
1. Apakah setiap fitur dalam dataset ini memiliki pengaruh pada model prediksi tumor payudara?
2. Model Machine Learning mana yang menyajikan model prediksi terbaik dalam menyelesaikan permasalahan diagnosis kanker payudara?

### Goals
1. Mengetahui fitur apa saja yang berpengaruh pada prediksi tumor payudara
2. Mengetahui model terbaik dalam Machine Learning untuk memprediksi ganas atau tidaknya kanker payudara

### Solution Statements
Untuk mencapai tujuan memprediksi Kanker Payudara, saya menggunakan 3 jenis model Klasifikasi Biner yang berbeda untuk memprediksi apakah diagnosisnya jinak (0) atau ganas (1). Ketigas algoritma tersebut cocok untuk memprediksi salah satu dari dua hasil yang mungkin terjadi. Penerapan SMOTE juga akan dilakukan untuk menangani data yang tidak seimbang. Algoritma untuk klasifikasi biner yang digunakan adalah sebagai berikut: <br>

- [Logistic Regression](https://www.datascienceinstitute.net/blog/binary-logistic-regression-an-introduction#:~:text=Binary%20logistic%20regression%20models%20the,or%20presence%20and%20so%20on.) <br>
Logistic Regression adalah hubungan antara satu set variabel independen; kategori atau kontinu, dan variabel dependen biner; seperti jinak atau ganas, mati atau bertahan hidup, begitu seterusnya dan seterusnya. Regresi logistik biasanya digunakan untuk masalah klasifikasi yang memprediksi nilai target berlabel 0 atau 1. Kurva regresi logistik adalah kurva sigmoid. Metrik kinerja umum untuk mengevaluasi model klasifikasi biner adalah Metrik kebingungan, skor akurasi; rasio prediksi yang benar, Presisi; proporsi prediksi positif adalah positif aktual, Sensitivitas (ingat); semakin tinggi skor penarikan, semakin baik model ML dalam mengidentifikasi positif, Spesifisitas; memprediksi dengan benar negatif dari negatif aktual, F-Score (F1-Score); menggabungkan presisi dan penarikan kembali. Dalam diagnosis medis,apa pun yang tidak memperhitungkan negatif palsu adalah serius, jadi [recall score](https://medium.com/javarevisited/evaluating-the-logistic-regression-ae2decf42d61) adalah ukuran yang lebih baik daripada presisi dalam hal ini.

- [Neural Network](https://medium.com/afblabs-data-science/a-simple-neural-networks-for-binary-classification-understanding-feed-forward-68c3c0659f78) <br>
Jaringan saraf adalah jaringan bentuk sederhana untuk mengklasifikasikan juga menekan ulang data variabel input agar sesuai dengan variabel aktual, disebut sebagai y atau variabel target. Nilai yang diprediksi kemudian ditingkatkan selama banyak iterasi yang disebut zaman dengan menghitung dan meminimalkan kehilangan kesalahan. Ada 3 lapisan berbeda di Neural Network; lapisan input, lapisan tersembunyi dan lapisan output. Dalam jaringan saraf, perpustakaan Keras umumnya digunakan dalam model Pembelajaran Mendalam. Untuk klasifikasi biner menggunakan Neural Network, Loss Function yang digunakan adalah Binary Crossentrophy dan tipe Activation Function adalah Sigmoid. Metrik Kinerja Evaluaion untuk klasifikasi biner di Neural Network adalah [Accuracy](https://towardsdatascience.com/the-explanation-you-need-on-binary-classification-metrics-321d280b590f). 

- [Support Vector Machine](https://medium.com/@24littledino/support-vector-machine-svm-in-python-fc3a4ffd25b6) <br>
Mesin Dukungan Vektor adalah seperangkat Metode Pembelajaran yang Dibimbing yang digunakan untuk masalah klasifikasi biner, deteksi regresi dan pencilan. Secara khusus, data Proyek SVM ke dimensi yang lebih tinggi, menemukan hyperplane optimal yang dapat memaksimalkan margin lunak, dan menggunakan hyperplane itu sebagai ambang batas untuk mengklasifikasikan titik data baru. Untuk mengevaluasi model SVM untuk tugas klasifikasi, lebih tepat menggunakan metrik khusus klasifikasi seperti akurasi, presisi, penarikan kembali, skor F1, dan area di bawah kurva ROC (ROC-AUC). Metrik ini memberikan wawasan tentang kemampuan model SVM untuk mengklasifikasikan instance dengan benar di berbagai kelas dan memperhitungkan karakteristik yang melekat pada tugas klasifikasi.

## Data Understanding
Dataset yang digunakan adalah dataset yang diambil dari Kaggle yakni [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/code). Dataset dalam bentuk Format CSV yang terdiri dari 357 Benign/Jinak dan 212 Malignant/Ganar. Total kesuluruhan dataset yang digunakan adalah 569 data dan terdiri dari 33 Fitur; 1 Fitur Kategorikal dan 32 Fitur Numerik. Satu Fitur Kategorikal akan dikonversi menjadi Binary Integer sebagai label target. Fitur-fitur dihitung dari gambar digital aspirasi jarum halus (FNA) dari massa payudara. Mereka menggambarkan karakteristik inti sel yang ada dalam gambar. Semua nilai fitur dicatat dengan empat digit signifikan dan tidak ada nilai yang hilang/missing values. Fitur yang diekstraksi adalah sebagai berikut: <br>

- ***Radius***:  Jari-jari dan masing-masing nukleus diukur dengan rata-rata panjang segmen garis radial yang ditentukan oleh centeroid ular dan masing-masing titik ular. (fitur numerik)
- ***Perimeter***: Jarak total antara titik ular merupakan perimeter nuklir. (fitur numerik)
- ***Area***: Area nuklir diukur hanya dengan menghitung jumlah piksel pada bagian dalam ular dan menambahkan satu-bantalan piksel dalam perimeter. (fitur numerik)
- ***Compactness***: Perimeter dan area digabungkan untuk memberikan ukuran kekompakan inti sel menggunakan rumus perimeter2/area. (fitur numerik)
- ***Smoothness***: Diukur dengan mengukur perbedaan antara panjang garis radial dan panjang rata-rata garis di sekitarnya. (fitur numerik)
- ***Concavity***: Tingkat keparahan bagian cekung dari kontur (fitur numerik)
- ***Concave points***: Jumlah bagian cekung dari kontur (fitur numerik)
- ***Symmetry***: Untuk mengukur simetri, sumbu utama, atau akor terpanjang melalui pusat, ditemukan. (fitur numerik)
- ***Fractal Dimension***: Diperkirakan menggunakan "Pendekatan Coastline" yang dijelaskan oleh Mandelbrot. (fitur numerik)
- ***Texture***: diukur dengan menemukan varians dari intensitas skala abu-abu dalam piksel komponen. (fitur numerik)
- ***Diagnostic***: jinak atau ganasnya suatu tumor sel payudara (fitur kategorikal)
    - Benign: Jinak
    - Malignant: Ganas

> Catatan Tambahan: _mean, _se (kesalahan standar) dan _worst (terbesar): rata-rata dari 3 nilai terbesar dari fitur ini dihitung untuk setiap gambar, menghasilkan 30 fitur. Misalnya: bidang 3 adalah jari-jari rata-rata, bidang 13 adalah jari-jari_se, bidang 23 adalah jari-jari terburuk.

 . | radius_mean | radius_se | radius_worst | 
 --- | --- | --- | --- | 
Definition | mean of distances from center to points on the perimeter | standard error for the mean of distances from center to points on the perimeter | "worst" or largest mean value for mean of distances from center to points on the perimeter | 
Values | 17.99 | 1.095| 25.38 | 


Untuk memahami sebuah data dengan memiliki jumlah yang banyak akan lebih efisien jika kita menggunakan yang disebut dengan visualisasi data. Dalam proyek ini, menggunakan beberapa visualisasi yang ada diikuti dengan pemeriksaan jenis data, missing values, data terduplikasi, outliers dan analisis statistik deskriptif.  

Periksa jenis data <br>
![df info](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/34570f7b-7e16-414e-89f6-0599f85d0f78)

Periksa Missing Values <br>
![df isna](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/e14838d5-5672-4119-95dd-fd14397d81d0)

Periksa Data Duplication <br>
![df duplicated](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/78819fb9-3ed0-47f7-97fa-da9328168a27)

Analisis Statistik Deskriptif <br>
![df describe](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/9b0f5c49-5927-4b31-adad-21dcd35e6f67)

Periksa Outliers <br>
Empat Contoh Fitur yang digunakan untuk Memeriksa Outlier:

| radius_mean | texture_mean | perimeter mean | area mean |
| :---: | :---: | :---: | :---: | 
| ![radius mean](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/319ae8e9-47d5-46df-b475-69291e915794)  | ![texture mean](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/2b529395-c8a8-4c5d-8dd4-68ab9de92e6a) |  ![perimeter mean](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/e3edbf37-30f9-4cca-88b8-06df171c8002)  | ![area mean](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/cac1e7b7-c579-4d9a-b860-67c326c7a144) |

Lakukan Univariate Analysis
![univariate analysis](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/ac1b6f89-6e0e-4a0f-a564-19ed276c37bc)

Lakukan Multivariate Analysis
![pairplot](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/0b1edff5-93de-4089-8fdc-1fe13e92836f)
<br>
![multivariate](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/06078f33-9f1a-4f04-849f-90fb6d9e97de)

## Data Preparation
Pada tahap ini, metode PCA, ubah tipe data fitur kategorikal menjadi bilangan biner integer, Metode IQR, SMOTE, dan Scalling Fitur akan dilakukan pada penelitian ini. Selain itu, dataset memang tidak seimbang; 357 Benign dan 212 Malignant sehingga SMOTE atau Teknik Over-sampling Minoritas Sintetis akan diterapkan. Menghapus outlier akan dilakukan juga dan diikuti oleh penskalaan fitur dengan normalisasi skor-z dimana hasil mean akan menjadi 0 dan standar deviasi 1. Ukuran data akan dipecah menjadi train set dan test set dengan rasio 80:20. Untuk memahami secara mendalam proses persiapan data adalah dengan melihat beberapa langkah ini berikut ini: <br>

1. Ubah "Diagnosis" Fitur tipe "objek" menjadi nilai "biner integer" 0 dan 1.
>Mengapa teknik ini perlu dilakukan?
>>Jawab: membutuhkan fitur input menjadi numerik sehingga algoritma dapat dilakukan. <br>
<img src="https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/a29b20da-b0f5-4b34-b066-2c2ed1e285d9"/> <img src="https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/89166fd5-dce2-499c-8a6d-3909cafa9db2"/> 

2. Hapus outliers menggunakan Metode IQR di semua Fitur. Kemudian, periksa bentuk data.
>Mengapa Metode IQR ini perlu dilakukan?
>>Jawaban: pencilan dapat meningkatkan varians dalam dataset dan metode Interquartile Range (IQR) adalah teknik yang kuat dan umum digunakan untuk mendeteksi dan menghapus pencilan juga.

![data shape after drop outliers](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/26534822-5f9f-451a-9efe-cbccc9b7c3eb) <br>
Dataset telah dibersihkan dan memiliki 398 sampel

3. Kurangi dimensi fitur radius_mean, perimeter_mean, area_mean, radius_worst, perimeter_worst, dan area_worst menggunakan PCA
>Mengapa perlu untuk mengurangi fitur-fitur menggunakan PCA untuk dilakukan?
>>Jawaban: Hasil pairplot menunjukkan keenam fitur tersebut memiliki korelasi yang tinggi sehingga dapat dikurangi menggunakan PCA yang membantu mengurangi kebisingan dan redundansi dalam dataset.

![dimension](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/bf28fcf9-1c18-4ecd-8284-fd15c807384a)

4. Memisahkan data menjadi Train Set and Test Set + SMOTE Untuk Penanganan Imbalanced data
>Mengapa perlu untuk menangani data yang tidak seimbang menggunakan SMOTE untuk dilakukan?
>>Jawaban: Untuk memaksimalkan akurasi keseluruhan dan meminimalkan MSE yang dapat menyesatkan ketika kelas tidak seimbang dan SMOTE (Teknik Over-sampling Minoritas Sintetis) adalah salah satu metode yang digunakan untuk mengatasi masalah ini.

![output smote](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/0c0b9474-ad9f-4f9a-9482-8648a8642ae4)

5. Feature Scalling menggunakan Z-Score Normalization
>Mengapa penskalaan fitur perlu dilakukan?
>>Jawab: Untuk memastikan semua fitur memiliki skala yang sama, biasanya antara 0 dan 1 atau sekitar rata-rata 0 dengan standar deviasi 1.

![standarisation](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/ed2b1602-7ea1-4089-a6ed-c9d7c9037585)

## Modelling
Untuk mengatasi masalah Klasifikasi Diagnosis Medis Biner ini, menerapkan Regresi Logistik, Jaringan Neural dan Mesin Vektor Pendukung Algoritma adalah model yang paling tepat dalam mengklasifikasikan apakah itu Benign (0) atau Malignant (1) dan memiliki akurasi yang bagus untuk prediksi. Berikut ini adalah penjelasan dari setiap tahap dalam setiap algoritma:

- [Logistic Regression Model:](https://medium.com/@akshayjain_757396/advantages-and-disadvantages-of-logistic-regression-in-machine-learning-a6a247e42b20) <br>
  Step 1. Import library <br>
  Step 2. Scale features (optional but often recommended)
  Step 3. Initialize and train logistic regression model before SMOTE <br>
  Step 4. Make predictions on the test set before SMOTE <br>
  Step 5. Evaluate performance before SMOTE <br>
  ![LogReg_Before Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/73a0b293-33a1-4855-ac9b-577a2adc33ac) <br>
  Step 6. Initialize and train logistic regression model after SMOTE <br>
  Step 7. Make predictions on the test set after SMOTE <br>
  Step 8. Evaluate performance after SMOTE <br>
  ![LogReg_After Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/77ddc2a4-4a7c-43eb-8046-0a56a6caa393) <br>
  Step 9. Plot Confusion Metrics Before and After SMOTE <br>
  ![LogReg Confusion Before and After smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/1d373926-f9db-4916-974a-127d1dd8c834) <br>
  Step 10. Plot Accuracy Before and After SMOTE <br>
  ![LogReg_Accuracy_Before and after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/63740f0e-86f4-433d-bc0c-3cb69da4c3bf) <br>

  > Kelebihan: <br>
  >> - Regresi Logistik berkinerja baik ketika dataset dapat dipisahkan secara linear <br>
  >> - Regresi Logistik tidak hanya memberikan ukuran seberapa relevan prediktor (ukuran efisien), tetapi juga arah asosiasi (positif atau negatif) <br>
  
  > Kekurangan: <br>
  >> - Batasan utama Regresi Logistik adalah asumsi linearitas antara variabel dependen dan variabel independen <br>
  >> - Jika jumlah pengamatan lebih rendah dari jumlah fitur, Regresi Logistik tidak boleh digunakan, jika tidak maka akan menyebabkan kelebihan  <br>



- [Neural Network Model](https://subscription.packtpub.com/book/data/9781788397872/1/ch01lvl1sec27/pros-and-cons-of-neural-networks): <br>
  Step 1. Import library <br>
  Step 2. Scale features (optional but often recommended) <br>
  Step 3. Define the neural network model architecture before SMOTE <br>
  Step 4. Compile the model <br>
  Step 5. Train the model before SMOTE <br>
  Step 6. Make predictions on the test set before SMOTE <br>
  Step 7. Compute and print confusion matrix and classification report before SMOTE <br>
  ![NN_Before Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/b5dbe9b3-47b7-438a-bd50-7aa594a65a3e) <br>
  Step 8. Define the neural network model architecture after SMOTE <br>
  Step 9. Compile the model <br>
  Step 10. Train the model after SMOTE <br>
  Step 11. Make predictions on the test set after SMOTE <br>
  Step 12. Compute and print confusion matrix and classification report after SMOTE <br>
  ![NN After Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/5dbf196c-9751-43e7-a87c-e7541b3d0b12) <br>
  Step 13. Plot confusion matrix before and after SMOTE <br>
  ![NN_ConfusionMetrics](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/436f2328-888d-4abd-8ed2-397e03db62ca) <br>
  Step 14. Plot Accuracy Before and After SMOTE <br>
  | Plot Accuracy 1 | Plot Accuracy 2 | 
  | :---: | :---: | 
  | ![NN_Accuracy Before after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/d88fa925-694f-47bf-a0f3-5edd86021790) | ![NN_Diagram Accuracy before after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/d7b4fb46-431c-46a7-9e85-fcd8128b0999) | 

  > Kelebihan: <br>
  >> - Neural Network dapat dilatih dengan sejumlah input dan lapisan <br>
  >> - Neural networks bekerja paling baik dengan lebih banyak titik data
  >> - Satu terlatih, prediksi cukup cepat
  
  >  Kekurangan: <br>
  >> - Neural networks adalah kotak hitam, artinya kita tidak bisa tahu bagaimana setiap variabel independen mempengaruhi variabel dependen <br>
  >> - Secara komputasi sangat mahal dan memakan waktu untuk berlatih dengan CPU tradisional  <br>
  >> - Neural networks sangat bergantung pada data pelatihan



- [Support Vector Machine Model:](https://scikit-learn.org/stable/modules/svm.html) <br>
  Step 1. Import library <br>
  Step 2. Initialize SVM classifier before SMOTE <br>
  Step 3. Train SVM classifier before SMOTE <br>
  Step 4. Make predictions on the test set before SMOTE <br>
  Step 5. Compute and print confusion matrix and classification report before SMOTE <br>
  ![SVM_Before smote Confusion](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/5ca15720-e31d-4c6c-a0be-6e59297cb226) <br>
  Step 6. Train SVM classifier after SMOTE <br>
  Step 7. Make predictions on the test set after SMOTE <br>
  Step 8. Compute and print confusion matrix and classification report after SMOTE <br>
  ![SVM_After SMOTE Confusion](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/071e71d2-8392-41c2-9f1a-0f4938a903e1) <br>
  Step 9. Plot Confusion Metrics Before and After SMOTE <br>
  ![Plot confusion metric SVM](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/e5a86814-f223-45ea-892e-22bda98e9af1) <br>
  Step 10. Plot Accuracy Before and After SMOTE <br>
  ![SVM Accuracy Before adn After SVM](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/0cee870f-92b6-4a87-87f0-d3ada2d971e9) <br>

  > Kelebihan: <br>
  >> - Efektif di Ruang Dimensi Tinggi <br>
  >> - Masih efektif dalam kasus di mana jumlah dimensi lebih besar dari jumlah sampel
  >> - Menggunakan subset poin pelatihan dalam fungsi keputusan / vektor pendukung, sehingga memori juga efisien
  >> - Serbaguna: fungsi kernel yang berbeda dapat ditentukan untuk fungsi keputusan.

  > Kekurangan: <br>
  >> - Jika jumlah fitur jauh lebih besar daripada jumlah sampel, hindari pas dalam memilih fungsi kernel dan istilah regularisasi sangat penting <br>
  >> - SVM tidak secara langsung memberikan perkiraan probabilitas, ini dihitung menggunakan validasi silang lima kali lipat yang mahal <br>

- Algoritma mana yang terbaik untuk prediksi? <br>
  | Logistic Regression | Neural Network | Support Vector Machine |
  | :---: | :---: | :---: | 
  | ![LogReg_Accuracy_Before and after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/051a8bc0-2c44-4e54-a377-18e6c0e9e5bb) | ![NN_Diagram Accuracy before after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/635bc08d-d78f-4b57-b6c2-c71549879242) | ![SVM Accuracy Before adn After SVM](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/ac82d963-2026-4273-bcdc-2ed9f84a5033) | <br>

  Algoritma terbaik untuk predicion Binary Integer Data dalam hal ini adalah ***Neural Network + SMOTE*** dengan akurasi 98%. Berdasarkan plot hasil akurasi antara NN Before dan After SMOTE, Neural Network dapat secara otomatis mempelajari fitur-fitur yang relevan dari data mentah dan memiliki fleksibilitas dalam berbagai arsitektur yang memungkinkan mereka untuk menangkap berbagai jenis pola dan struktur dalam data, membuatnya mudah beradaptasi dengan beragam tugas klasifikasi di berbagai domain. Dalam model ini, mendorong Fungsi RelU Aktivasi di lapisan input, lapisan tersembunyi dan Fungsi Acitvation Sigmoid dalam lapisan Output adalah alasan utama mengapa Neural Network adalah algoritma yang paling tepat untuk mengklasifikasikan Target yang Diberi Label ke dalam data ineteger biner. <br>

   ![NN_Accuracy Before after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/6c8ead2e-918c-4eb4-ba1f-c364b1270279)

## Evaluation
- Evaluasi metrik apa yang digunakan dalam proyek ini dan cara kerjanya? <br> 
Evaluasi metrik yang digunakan adalah [Confusion Metrics, Accuracy, Precision, Sensitivity(Recall), Specificity and F1-Score](https://medium.com/javarevisited/evaluating-the-logistic-regression-ae2decf42d61). Metrik kinerja tersebut biasanya digunakan untuk mengevaluasi Model Klasifikasi Biner. Mari kita tentukan setiap metrik dan cara kerjanya di bawah ini:<br>
1. Confusion Metric: <br>
   Ringkasan tabular dari tingkat prediksi Benar / Salah dan Positif / Negatif. Hal ini memungkinkan untuk menghitung berbagai metrik kinerja. <br>

   ![Basic-Confusion-matrix](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/2fd76ef0-c24e-4ba3-ae8a-0959bf5b1e9a)
   
2. Accuracy: <br>
mengukur rasio prediksi yang benar dari semua hasil yang diprediksi. <br>

   ![accurcay score measuring](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/dd2b037c-cb5d-4cbd-a49c-547d67154a16)

3. Precision: <br>
mengukur berapa proporsi prediksi positif yang sebenarnya positif. Skor presisi berguna untuk keberhasilan prediksi ketika kelas sangat tidak seimbang dan ketika secara signifikan hemat biaya untuk mengidentifikasi semua contoh positif tanpa positif palsu.

<br>

   ![precision formula](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/8d654914-c0eb-4047-95f6-e72aa5e9efbb)

4. Sensitivity/Recall: <br>
mewakili kemampuan model untuk memprediksi dengan benar positif dari positif aktual. Semakin tinggi skor penarikan, semakin baik model pembelajaran mesin dalam mengidentifikasi hal-hal positif. <br>
   ![recall formula](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/8a47bebf-2e71-4b39-8147-8d90b0ff3673)

5. Specificity: <br>
mewakili kemampuan model untuk memprediksi dengan benar negatif dari negatif aktual. Semakin tinggi skor spesifisitas, semakin baik model pembelajaran mesin dalam mengidentifikasi negatif. <br>

   ![specificity](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/0a7141f5-3bf6-4c89-bfcf-51010985190a)

6. F1 Score: <br>
menggabungkan ketepatan dan penarikan kembali model, dan itu adalah rata-rata harmonik dari presisi dan penarikan. Ini sering digunakan ketika data tidak seimbang. <br>

   ![f1 score](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/4a047010-c742-47c0-a375-015bf2b223c8)

 
- Penjelasan Hasil Proyek Evaluasi Metrik! <br>
  | X | Logistic Regression | Neural Network |  Support Vector Machine | 
  | :---: | :---: | :---: | :---: | 
  | Confusion Metrics Before SMOTE | ![LogReg_Before Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/4d073946-e2b4-410a-8bb5-749e25f9e604) | ![NN_Before Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/ffcb28c5-012c-4f6e-a493-563209fc2b98)| ![SVM_Before smote Confusion](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/29275888-f04f-41a2-a6a8-1a9d46e53f59) | 
  | Confusion Metrics After SMOTE | ![LogReg_After Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/8321ccc2-2688-4992-8264-727ddec9b2ea) |![NN After Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/ddc13079-5354-42ee-977f-c813b8331355) | ![SVM_After SMOTE Confusion](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/e5d4a8a8-b87f-4782-8a2c-f8091f30e178)|
  | Plot Accuracy |  ![LogReg_Accuracy_Before and after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/051a8bc0-2c44-4e54-a377-18e6c0e9e5bb) | ![NN_Diagram Accuracy before after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/635bc08d-d78f-4b57-b6c2-c71549879242) | ![SVM Accuracy Before adn After SVM](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/ac82d963-2026-4273-bcdc-2ed9f84a5033) |

 Pada perbandingan matriks kebingungan dan nilai akurasi hasil di atas, Neural Network dengan SMOTE memiliki skor akurasi tertinggi dan matriks kebingungan yang hebat sehingga algoritma ini adalah model yang paling cocok dan tepat untuk prediksi Diagnosis Tumor Payudara menggunakan Fitur Phenotip Sel-Nuklir Kuantitatif. Fleksibilitas Arsitektur Jaringan Neural memungkinkan untuk menangkap berbagai jenis pola/struktur dalam dataset dan pelajari fitur numerik yang relevan untuk memprediksi Binary Integer Data apakah itu Benign (0) atau Malignant (1).

## Conclusion 
1. Tidak semua fitur berdampak pada algoritma dalam prediksi model.
2. Neural Network dengan menangani data yang tidak seimbang menggunakan SMOTE, memberikan akurasi terbesar sehingga algoritma ini adalah model terbaik untuk memprediksi Diagnosis Tumor Payudara.
3. Ya, SMOTE memengaruhi akurasi tinggi model.
