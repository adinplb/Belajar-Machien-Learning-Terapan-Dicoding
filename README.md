# Laporan Proyek Machine Learning - Muhammad Adin Palimbani

## Domain Proyek
Kemajuan yang signifikan dalam penelitian kanker selama beberapa dekade terakhir telah dilakukan dengan munculnya teknologi baru di bidang kedokteran. Para ilmuwan telah melakukan pendekatan baru dengan metode yang berbeda untuk prediksi awal hasil pengobatan kanker terutama Kanker Payudara. Salah satu contoh pendekatan yang diterapkan adalah tren yang berkembang pada Machine Learning. Namun, [masalah umum dalam beberapa penelitian](https://www.sciencedirect.com/science/article/pii/S1877050921014629) adalah kurangnya validasi eksternal atau pengujian mengenai kinerja prediktif model mereka dan juga menangani data yang tidak seimbang. Ini dapat menyebabkan model prediksi pada kanker yang salah dan kegagalan sistem pada tahap produksi. Model prediksi akurat dari hasil penyakit sangat tergantung pada data medis pasien. Data medis berisi kondisi detail pasien dan diagnosis yang menyimpan data yang tidak perlu dan saling terkait. Dalam beberapa penelitian, para ilmuwan telah membuktikan bahwa pendekatan yang berkaitan dengan karakteristik genom memberikan hasil yang menjanjikan untuk deteksi dan identifikasi kanker, misalnya, gambar digital dari aspirat jarum halus (FNA) pada massa payudara yang mewakili karakteristik sel nukleus dalam Tumor Payudara. Namun, metode ini menderita sensitivitas rendah mengenai penggunaannya dalam skrining pada tahap awal dan kesulitan untuk menentukan jinak dari tumor ganas. Ini adalah alasan mengapa masalah model kinerja prediktif kanker perlu diselesaikan untuk mencegah prediksi yang salah dan kegagalan sistem dalam mendiagnosis apakah itu ganas atau jinak melalui Ekstraksi Fitur Sel Nukleus.

## Business Understanding
Diagnosis Tumor Payudara telah dilakukan dengan metode [Fine Needle Aspiration (FNA)](https://cancer.ca/en/treatments/tests-and-procedures/fine-needle-aspiration-fna); jenis biopsi yang menggunakan jarum dan jarum suntik yang sangat tipis untuk menghilangkan sampel sel, jaringan atau cairan dari area abnormal atau benjolan dalam tubuh. FNA telah berhasil mendiagnosis karakteristik sel nukleus dan menjadi fitur yang menunjukkan kemungkinan keganasan yang lebih tinggi. Sistem diagnostik visi komputer mengekstrak 10 fitur berbeda dari batas inti sel yang dihasilkan ular. Fitur yang diekstraksi dimodelkan secara numerik yang terdiri dari Radius, Perimeter, Area, Kekompakan, Kelancaran, rasa hormat, Poin Cekung, Simetri, Dimensi Fratal dan Tekstur. Untuk menjawab masalah ini, predictive analytics dengan supervised machine learning diharapkan dapat memprediksi masalah tersebut dan mendapatkan solusi yang terbaik dengan menggunakan model machine learning.

### Problem Statement
Berikut adalah problem statement dari proyek ini:
- Apakah setiap fitur dalam dataset ini memiliki pengaruh pada model prediksi tumor payudara?
- Model Machine Learning mana yang menyajikan model prediksi terbaik dalam menyelesaikan permasalahan diagnosis kanker payudara?

### Goals
- Mengetahui fitur apa saja yang berpengaruh pada prediksi tumor payudara
- Mengetahui model terbaik dalam Machine Learning untuk memprediksi ganas atau tidaknya kanker payudara

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
Pada tahap ini, metode PCA, ubah tipe data fitur kategorikal menjadi bilangan biner integer, Metode IQR, SMOTE, dan Scalling Fitur akan dilakukan pada penelitian ini. Selain itu, dataset memang tidak seimbang; 357 Benign dan 212 Malignant sehingga SMOTE atau Teknik Over-sampling Minoritas Sintetis akan diterapkan. Menghapus outlier akan dilakukan juga dan diikuti oleh penskalaan fitur dengan normalisasi skor-z dimana hasil mean akan menjadi 0 dan standar deviasi 1. Ukuran data akan dipecah menjadi train set dan test set dengan rasio 80:20. Berikut tahapan persiapan data:
- Ubah "Diagnosis" Fitur tipe "objek" menjadi nilai "biner integer" 0 dan 1. Tujuan dialkukan ini ialah model membutuhkan fitur input dalam bentuk numerik sehingga algoritma dapat diproses
- Hapus outliers menggunakan Metode IQR di semua Fitur dan periksa bentuk data sehingga didapatkan Dataset bersih sebanyak 398 sampel. Metode ini dilakukan dengan tujuan agar pencilan dapat meningkatkan varians dalam bentuk dataset sehingga metode IQR merupakan teknik yang kuat dalam mendeteksi dan menghapus pencilan. 
- Kurangi dimensi fitur radius_mean, perimeter_mean, area_mean, radius_worst, perimeter_worst, dan area_worst menggunakan PCA. Hasil pairplot menunjukkan keenam fitur numerik tersebut memiliki korelasi yang tinggi sehingga dapat dikurangi menggunakan PCA yang membantu mengurangi noise dan redudancy dalam dataset
- Kemudian, memisahkan data menjadi Train Set and Test Set + SMOTE untuk Penanganan Imbalanced data dan hanya dilakukan pada data train saja. Tujuan dilakukannya SMOTE guna memaksimalkan akurasi keseluruhan dan mempertahankan keaslian data saat melakukan latih data. 
- Melakukan Feature Scalling menggunakan Z-Score Normalization. Tujuan dari teknik ini ialah memastikan semua fitur memiliki skala yang sama sehingga hasil mean menjadi 0 dan standar deviasinya menjadi 1. 

## Modelling
Untuk mengatasi masalah Klasifikasi Diagnosis Medis Biner ini, menerapkan Regresi Logistik, Jaringan Neural dan Mesin Vektor Pendukung Algoritma adalah model yang paling tepat dalam mengklasifikasikan apakah itu Benign (0) atau Malignant (1) dan memiliki akurasi yang bagus untuk prediksi. Algortma yang digunakan adalah Logistic Regression, Neural Network dan Support Vector Machine. 
- Logistic Regression: memiliki kelebihan dalam kinerja yang dapat dipisahkan secara linear dan memberikan ukuran seberapa relevan prediktor (ukuran efisien) serta arah asosiasi (positif atau negatif). Kekurangan dari logistic regression adalah Batasan utama Regresi Logistik adalah asumsi linearitas antara variabel dependen dan variabel independen. Tidak hanya itu, Jika jumlah pengamatan lebih rendah dari jumlah fitur, Logistic Regression tidak boleh digunakan, jika tidak maka akan menyebabkan kelebihan.
- Neural Network dimana memiliki 1 input layer dengan Activation RelU, 2 Hidden layer dengan activation RelU dan Output layer dengan Activation Sigmoid Function. Kelebihan dari neural network adalah dapat dilatih dengan sejumlah input dan lapisan, bekerja paling baik dengan lebih banyak titik data dan Satu terlatih, prediksi cukup cepat. Namun, kekurangan dari Neural Network adalah tidak bisa tahu bagaimana setiap variabel independen mempengaruhi variabel dependen dan secara komputasi sangat mahal serta memakan waktu untuk berlatih dengan CPU tradisional.
- Support Vector Machine dengan kernel RBF. Kelebihan dari algoritma ini adalah efektif di Ruang Dimensi Tinggi, masih efektif dalam kasus di mana jumlah dimensi lebih besar dari jumlah sampel, menggunakan subset poin pelatihan dalam fungsi keputusan / vektor pendukung, sehingga memori juga efisien dam bersifat versatile. Kekurangan dari Support Vector Machine adalah jika jumlah fitur jauh lebih besar daripada jumlah sampel, hindari pas dalam memilih fungsi kernel dan istilah regularisasi sangat penting. Selain itu, SVM tidak secara langsung memberikan perkiraan probabilitas, ini dihitung menggunakan validasi silang lima kali lipat yang mahal. <br>
- Berdasarkan hasil dan evaluasi pelatihan model, algoritma terbaik yang direkomendasikan dalam melakukan prediksi kanker payudara adalah Neural Network diikuti dengan penanganan data tidak seimbang menggunakan SMOTE.

## Evaluation
Evaluasi metrik yang digunakan adalah Confusion Metrics, Accuracy, Precision, Sensitivity(Recall), Specificity and F1-Score. Metrik kinerja tersebut biasanya digunakan untuk mengevaluasi Model Klasifikasi Biner. Confusion Metric adalah ringkasan tabular dari tingkat prediksi Benar / Salah dan Positif / Negatif. Hal ini memungkinkan untuk menghitung berbagai metrik kinerja. Accuracy bekerja dengan mengukur rasio prediksi yang benar dari semua hasil yang diprediksi. Nilai Precision bekerja dengan mengukur berapa proporsi prediksi positif yang sebenarnya positif. Skor presisi berguna untuk keberhasilan prediksi ketika kelas sangat tidak seimbang dan ketika secara signifikan hemat biaya untuk mengidentifikasi semua contoh positif tanpa positif palsu. Sensitivity/Recall yaitu mewakili kemampuan model untuk memprediksi dengan benar positif dari positif aktual. Semakin tinggi skor penarikan, semakin baik model pembelajaran mesin dalam mengidentifikasi hal-hal positif. Specificity adalah mewakili kemampuan model untuk memprediksi dengan benar negatif dari negatif aktual. Semakin tinggi skor spesifisitas, semakin baik model pembelajaran mesin dalam mengidentifikasi negatif.F1 Score bekerja dengan menggabungkan ketepatan dan penarikan kembali model, dan itu adalah rata-rata harmonik dari presisi dan penarikan. Ini sering digunakan ketika data tidak seimbang. 

|   | Accuracy  | Precision  |  Recall |  F1-Score |
|---|---|---|---|---|
| Logistic Regression  | 0.81  | 0.86  | 0.81  | 0.82  |
|  Logistic Regression +SMOTE | 0.95  |  0.95 | 0.95  | 0.95  |
|  Neural Network | 0.81  | 0.86  | 0.81  | 0.82  |
| Neural Network + SMOTE  |  0.97 | 0.98  | 0.97  | 0.97  |
|  SVM | 0.69  | 0.47  |  0.69 | 0.56  |
|  SVM +SMOTE | 0.95  | 0.95  | 0.95 | 0.95  |

Hasil evaluasi metric menunjukkan bahwa Neural Network dengan SMOTE memiliki skor akurasi tertinggi dan matriks kebingungan yang hebat sehingga algoritma ini adalah model yang paling cocok dan tepat untuk prediksi Diagnosis Tumor Payudara menggunakan Fitur Phenotip Sel-Nuklir Kuantitatif. Fleksibilitas Arsitektur Jaringan Neural memungkinkan untuk menangkap berbagai jenis pola/struktur dalam dataset dan pelajari fitur numerik yang relevan untuk memprediksi Binary Integer Data apakah itu Benign (0) atau Malignant (1). Hasil pemodelan menunjukkan bahwa algoritma terbaik untuk prediksi Binary Integer Data yaitu ***Neural Network + SMOTE*** dengan akurasi 98%. Berdasarkan plot hasil akurasi antara NN Before dan After SMOTE, Neural Network dapat secara otomatis mempelajari fitur-fitur yang relevan dari data mentah dan memiliki fleksibilitas dalam berbagai arsitektur yang memungkinkan mereka untuk menangkap berbagai jenis pola dan struktur dalam data, membuatnya mudah beradaptasi dengan beragam tugas klasifikasi di berbagai domain. Dalam model ini, mendorong Fungsi RelU Aktivasi di lapisan input, lapisan tersembunyi dan Fungsi Acitvation Sigmoid dalam lapisan Output adalah alasan utama mengapa Neural Network adalah algoritma yang paling tepat untuk mengklasifikasikan Target yang Diberi Label ke dalam data ineteger biner.

  | Logistic Regression | Neural Network | Support Vector Machine |
  | :---: | :---: | :---: | 
  | ![LogReg_Accuracy_Before and after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/051a8bc0-2c44-4e54-a377-18e6c0e9e5bb) | ![NN_Diagram Accuracy before after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/635bc08d-d78f-4b57-b6c2-c71549879242) | ![SVM Accuracy Before adn After SVM](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/ac82d963-2026-4273-bcdc-2ed9f84a5033) | <br>

 <br>

   ![NN_Accuracy Before after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/6c8ead2e-918c-4eb4-ba1f-c364b1270279)

## Conclusion 
Kesimpulan yang didapat dalam memprediksi biaya asuransi pada proyek ini adalah sebagai berikut:
- Tidak semua fitur berdampak pada algoritma dalam model prediksi tumor payudara
- Neural Network dengan menangani data yang tidak seimbang menggunakan SMOTE, memberikan akurasi terbesar sehingga algoritma ini adalah model prediksi terbaik untuk menyelesaikan permasalahan diagnosis kanker payudara
