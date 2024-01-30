# Laporan Proyek Machine Learning - Muhammad Adin Palimbani

## Project Domain 
1. Fokus Isu? <br>
Kemajuan yang signifikan dalam penelitian kanker selama beberapa dekade terakhir telah dilakukan dengan munculnya teknologi baru di bidang kedokteran. Para ilmuwan telah melakukan pendekatan baru dengan metode yang berbeda untuk prediksi awal hasil pengobatan kanker terutama Kanker Payudara. Salah satu contoh pendekatan yang diterapkan adalah tren yang berkembang pada Teknik Pembelajaran Mesin. Namun, masalah umum dalam beberapa penelitian adalah kurangnya validasi eksternal atau pengujian mengenai kinerja prediktif model mereka dan juga menangani data yang tidak seimbang. Ini dapat menyebabkan model prediksi yang salah dan kegagalan sistem pada tahap produksi

2. Mengapa masalah ini perlu diselesaikan?? <br>
Model prediksi akurat dari hasil penyakit sangat tergantung pada data medis pasien. Data medis berisi kondisi detail pasien dan diagnosis yang menyimpan data yang tidak perlu dan saling terkait. Data tersebut adalah data dimensi tinggi juga khususnya integrasi data campuran klinis dan genom. Dalam beberapa penelitian, para ilmuwan telah membuktikan bahwa pendekatan yang berkaitan dengan karakteristik genom memberikan hasil yang menjanjikan untuk deteksi dan identifikasi kanker, misalnya, gambar digital dari aspirat jarum halus (FNA) dari massa payudara yang mewakili karakteristik nuklir sel dalam Tumor Payudara. Namun, metode ini menderita sensitivitas rendah mengenai penggunaannya dalam skrining pada tahap awal dan kesulitan untuk menentukan jinak dari tumor ganas.Ini adalah alasan mengapa masalah model kinerja prediktif kanker perlu diselesaikan untuk mencegah prediksi yang salah dan kegagalan sistem. 

3. Bagaimana cara mengatasi masalah ini? <br>
Teknik pemrosesan gambar interaktif, bersama dengan pengklasifikasi induktif berbasis pemrograman linier, telah digunakan untuk merambat sistem yang sangat akurat untuk diagnosis Tumor Payudara. Sebagian kecil dari Fine Needle Aspirate Slide (FNA) dipilih dan didigitalkan. Gambar digital dari FNA dari massa Payudara menggambarkan chracteristics dari inti sel yang ada dalam gambar. Itu dihitung dan menjadi fitur untuk penelitian ini. Jadi kita bisa mendiagnosis apakah itu ganas atau jinak melalui Ekstraksi Fitur Nuklir.

4. Referensi Penelitian Terkait dari Sumber yang Dapat Dipercaya:: <br> 
[Nuclear Feature Extraction For Breast Tumor Diagnosis](https://minds.wisconsin.edu/bitstream/handle/1793/59692/TR1131.pdf;jsessionid=0449D8C1D78CAAB2BF57B76AABE87312?sequence=1). <br>
[Prediction of parameters of liver tumor using feature extraction and supervised function](https://www.sciencedirect.com/science/article/pii/S2665917422000204). <br>
[Machine Learning Algorithms For Breast Cancer Prediction And Diagnosis](https://www.sciencedirect.com/science/article/pii/S1877050921014629). <br>
[Quantitative nuclear phenotype signatures predict nodal disease in oral squamous cell carcinoma](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8568158/). <br>

## Business Understanding
Diagnosis Tumor Payudara telah dilakukan oleh [Fine Needle Aspiration (FNA)](https://cancer.ca/en/treatments/tests-and-procedures/fine-needle-aspiration-fna); jenis biopsi yang menggunakan jarum dan jarum suntik yang sangat tipis untuk menghilangkan sampel sel, jaringan atau cairan dari area abnormal atau benjolan dalam tubuh. FNA telah berhasil mendiagnosis fenotip nuklir sel dan menjadi fitur yang menunjukkan kemungkinan keganasan yang lebih tinggi. Sistem diagnostik visi komputer mengekstrak 10 fitur berbeda dari batas inti sel yang dihasilkan ular. Fitur yang diekstraksi dimodelkan secara numerik yang terdiri dari Radius, Perimeter, Area, Kekompakan, Kelancaran, rasa hormat, Poin Cekung, Simetri, Dimensi Fratal dan Tekstur. Selain itu, ada diagnosis fitur tumor payudara yang mewakili ganas atau bening sehingga dalam proyek ini ada target berlabel untuk memprediksi apakah itu tumor jinak atau ganas. Model Supervision Machine Learning cocok untuk menyelesaikan masalah ini dengan menggunakan fenotipe nuklir sel kuantitatif Breast Tumor dan diikuti dengan menangani data yang tidak seimbang.

### Problem Statement
1. Apakah setiap fitur dalam dataset ini memiliki pengaruh pada model prediksi tumor payudara?
2. Model Machine Learning mana yang menyajikan model prediksi terbaik dan dapat menyelesaikan masalah?
3. Bagaimana cara menangani data yang tidak seimbang memberikan pengaruh pada model prediksi?

### Goals
1. Temukan fitur yang berpengaruh pada prediksi tumor payudara
2. Temukan Model Pembelajaran Mesin terbaik yang mungkin bisa menyelesaikan masalah
3. Temukan pengaruh SMOTE dalam menangani data yang tidak seimbang untuk prediksi model

### Solution Statements
Untuk mencapai Prediksi Kanker Payudara yang baik, menggunakan 3 jenis model Klasifikasi Biner yang berbeda untuk memprediksi apakah diagnosisnya jinak (0) atau ganas (1) dalam Algoritma Pembelajaran Mesin yang Dibimbing. Algoritma tersebut cocok untuk memprediksi salah satu dari dua hasil yang mungkin. Sebagai tambahan, SMOTE akan diterapkan juga untuk menangani data yang tidak seimbang. Algoritma untuk klasifikasi biner adalah sebagai berikut: <br>

- [Binary Logistic Regression](https://www.datascienceinstitute.net/blog/binary-logistic-regression-an-introduction#:~:text=Binary%20logistic%20regression%20models%20the,or%20presence%20and%20so%20on.) <br>
Binary Logistic Regression adalah hubungan antara satu set variabel independen; kategori atau kontinu, dan variabel dependen biner; seperti jinak atau ganas, mati atau bertahan hidup, begitu seterusnya dan seterusnya. Regresi logistik biasanya digunakan untuk masalah klasifikasi yang memprediksi nilai target berlabel 0 atau 1. Kurva regresi logistik adalah kurva sigmoid. Metrik kinerja umum untuk mengevaluasi model klasifikasi biner adalah Metrik kebingungan, skor akurasi; rasio prediksi yang benar, Presisi; proporsi prediksi positif adalah positif aktual, Sensitivitas (ingat); semakin tinggi skor penarikan, semakin baik model ML dalam mengidentifikasi positif, Spesifisitas; memprediksi dengan benar negatif dari negatif aktual, F-Score (F1-Score); menggabungkan presisi dan penarikan kembali. Dalam diagnosis medis,apa pun yang tidak memperhitungkan negatif palsu adalah serius, jadi [recall score](https://medium.com/javarevisited/evaluating-the-logistic-regression-ae2decf42d61) adalah ukuran yang lebih baik daripada presisi dalam hal ini.

- [Neural Network](https://medium.com/afblabs-data-science/a-simple-neural-networks-for-binary-classification-understanding-feed-forward-68c3c0659f78) <br>
Jaringan saraf adalah jaringan bentuk sederhana untuk mengklasifikasikan juga menekan ulang data variabel input agar sesuai dengan variabel aktual, disebut sebagai y atau variabel target. Nilai yang diprediksi kemudian ditingkatkan selama banyak iterasi yang disebut zaman dengan menghitung dan meminimalkan kehilangan kesalahan. Ada 3 lapisan berbeda di Neural Network; lapisan input, lapisan tersembunyi dan lapisan output. Dalam jaringan saraf, perpustakaan Keras umumnya digunakan dalam model Pembelajaran Mendalam. Untuk klasifikasi biner menggunakan Neural Network, Loss Function yang digunakan adalah Binary Crossentrophy dan tipe Activation Function adalah Sigmoid. Metrik Kinerja Evaluaion untuk klasifikasi biner di Neural Network adalah [Accuracy](https://towardsdatascience.com/the-explanation-you-need-on-binary-classification-metrics-321d280b590f). 

- [Support Vector Machine](https://medium.com/@24littledino/support-vector-machine-svm-in-python-fc3a4ffd25b6) <br>
Mesin Dukungan Vektor adalah seperangkat Metode Pembelajaran yang Dibimbing yang digunakan untuk masalah klasifikasi biner, deteksi regresi dan pencilan. Secara khusus, data Proyek SVM ke dimensi yang lebih tinggi, menemukan hyperplane optimal yang dapat memaksimalkan margin lunak, dan menggunakan hyperplane itu sebagai ambang batas untuk mengklasifikasikan titik data baru. Untuk mengevaluasi model SVM untuk tugas klasifikasi, lebih tepat menggunakan metrik khusus klasifikasi seperti akurasi, presisi, penarikan kembali, skor F1, dan area di bawah kurva ROC (ROC-AUC). Metrik ini memberikan wawasan tentang kemampuan model SVM untuk mengklasifikasikan instance dengan benar di berbagai kelas dan memperhitungkan karakteristik yang melekat pada tugas klasifikasi.

## Data Understanding
Dataset yang digunakan dikumpulkan dari Kaggle. Ini [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/code). Dataset dalam Format File CSV yang total kontribusi kelasnya adalah 357 Benign dan 212 Malignant. Dataset terdiri dari 569 baris dan 33 Fitur; 1 Fitur Kategorikal dan 32 Fitur Numerik. Satu Fitur Kategorikal akan dikonversi menjadi Binary Integer sebagai Target Labeled. Fitur-fitur dihitung dari gambar digital aspirasi jarum halus (FNA) dari massa payudara. Mereka menggambarkan karakteristik inti sel yang ada dalam gambar. Semua nilai fitur dicatat dengan empat digit signifikan dan tidak ada nilai yang hilang. Fitur yang diekstraksi adalah sebagai berikut: <br>

1. ***Radius***:  Jari-jari dan masing-masing nukleus diukur dengan rata-rata panjang segmen garis radial yang ditentukan oleh centeroid ular dan masing-masing titik ular.
2. ***Perimeter***: Jarak total antara titik ular merupakan perimeter nuklir.
3. ***Area***: Area nuklir diukur hanya dengan menghitung jumlah piksel pada bagian dalam ular dan menambahkan satu-bantalan piksel dalam perimeter.
4. ***Compactness***: Perimeter dan area digabungkan untuk memberikan ukuran kekompakan inti sel menggunakan rumus perimeter2/area.
5. ***Smoothness***: Diukur dengan mengukur perbedaan antara panjang garis radial dan panjang rata-rata garis di sekitarnya.
6. ***Concavity***: Tingkat keparahan bagian cekung dari kontur
7. ***Concave points***: Jumlah bagian cekung dari kontur
8. ***Symmetry***: Untuk mengukur simetri, sumbu utama, atau akor terpanjang melalui pusat, ditemukan.
9. ***Fractal Dimension***: Diperkirakan menggunakan "Pendekatan Coastline" yang dijelaskan oleh Mandelbrot.
10. ***Texture***: diukur dengan menemukan varians dari intensitas skala abu-abu dalam piksel komponen.
11. ***Diagnostic***: Jinak dan ganas

> Catatan Tambahan: _mean, _se (kesalahan standar) dan _worst (terbesar): rata-rata dari 3 nilai terbesar dari fitur ini dihitung untuk setiap gambar, menghasilkan 30 fitur. Misalnya: bidang 3 adalah jari-jari rata-rata, bidang 13 adalah jari-jari_se, bidang 23 adalah jari-jari terburuk.

 . | radius_mean | radius_se | radius_worst | 
 --- | --- | --- | --- | 
Definition | mean of distances from center to points on the perimeter | standard error for the mean of distances from center to points on the perimeter | "worst" or largest mean value for mean of distances from center to points on the perimeter | 
Values | 17.99 | 1.095| 25.38 | 

### Exploratory Data Analysis and Visualization
#### Periksa Jenis Data
![df info](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/34570f7b-7e16-414e-89f6-0599f85d0f78)

#### Periksa Missing Values
![df isna](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/e14838d5-5672-4119-95dd-fd14397d81d0)

#### Periksa Data Duplication
![df duplicated](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/78819fb9-3ed0-47f7-97fa-da9328168a27)

#### Analisis Statistik Deskriptif
![df describe](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/9b0f5c49-5927-4b31-adad-21dcd35e6f67)

#### Periksa Outliers
Empat Contoh Fitur yang digunakan untuk Memeriksa Outlier:

| radius_mean | texture_mean | perimeter mean | area mean |
| :---: | :---: | :---: | :---: | 
| ![radius mean](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/319ae8e9-47d5-46df-b475-69291e915794)  | ![texture mean](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/2b529395-c8a8-4c5d-8dd4-68ab9de92e6a) |  ![perimeter mean](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/e3edbf37-30f9-4cca-88b8-06df171c8002)  | ![area mean](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/cac1e7b7-c579-4d9a-b860-67c326c7a144) |

#### Univariate Analysis
![univariate analysis](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/ac1b6f89-6e0e-4a0f-a564-19ed276c37bc)

#### Multivariate Analysis
![pairplot](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/0b1edff5-93de-4089-8fdc-1fe13e92836f)
<br>
![multivariate](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/06078f33-9f1a-4f04-849f-90fb6d9e97de)

## Data Preparation
Pada tahap ini, pengurangan feaures PCA, ubah tipe berlabel target menjadi bilangan bulat biner, Metode IQR, SMOTE, dan Scalling Fitur adalah teknik approriate untuk jenis dataset ini. Selain itu, kontribusi kelas dalam dataset memang tidak seimbang; 357 Benign dan 212 Malignant sehingga SMOTE atau Teknik Over-sampling Minoritas Sintetis akan diterapkan. Menghapus outlier akan dilakukan juga dan diikuti oleh penskalaan fitur atau normalisasi skor-z di mana mereka memiliki rata-rata 0 dan standar deviasi 1. Ukuran data akan dipecah menjadi set kereta dan set tes dengan rasio 80:20. Untuk memahami secara mendalam seluk beluk persiapan data adalah dengan melihat beberapa langkah ini: <br>

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
  
  Step 5. Evaluate performance before SMOTE

  ![LogReg_Before Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/73a0b293-33a1-4855-ac9b-577a2adc33ac)

  Step 6. Initialize and train logistic regression model after SMOTE <br>
  
  Step 7. Make predictions on the test set after SMOTE
  
  Step 8. Evaluate performance after SMOTE
  
  ![LogReg_After Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/77ddc2a4-4a7c-43eb-8046-0a56a6caa393)

  Step 9. Plot Confusion Metrics Before and After SMOTE
  ```ruby
  # Compute confusion matrices before and after SMOTE
  confusion_matrix_before_smote = confusion_matrix(y_test, y_pred_before_smote)
  confusion_matrix_after_smote = confusion_matrix(y_test, y_pred_after_smote)

  # Plot confusion matrices
  fig, axes = plt.subplots(1, 2, figsize=(12, 6))

  # Plot confusion matrix before SMOTE
  axes[0].imshow(confusion_matrix_before_smote, cmap=plt.cm.Blues, interpolation='nearest')
  axes[0].set_title('Confusion Matrix Before SMOTE')
  axes[0].set_xticks([0, 1])
  axes[0].set_yticks([0, 1])
  axes[0].set_xlabel('Predicted Label')
  axes[0].set_ylabel('True Label')
  for i in range(2):
      for j in range(2):
          axes[0].text(j, i, str(confusion_matrix_before_smote[i, j]),
                       horizontalalignment='center', verticalalignment='center',   color='white')

  # Plot confusion matrix after SMOTE
  axes[1].imshow(confusion_matrix_after_smote, cmap=plt.cm.Blues, interpolation='nearest')
  axes[1].set_title('Confusion Matrix After SMOTE')
  axes[1].set_xticks([0, 1])
  axes[1].set_yticks([0, 1])
  axes[1].set_xlabel('Predicted Label')
  axes[1].set_ylabel('True Label')
  for i in range(2):
      for j in range(2):
          axes[1].text(j, i, str(confusion_matrix_after_smote[i, j]),
                       horizontalalignment='center', verticalalignment='center', color='white')

  plt.tight_layout()
  plt.show()
  ```

  ![LogReg Confusion Before and After smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/1d373926-f9db-4916-974a-127d1dd8c834)

  Step 10. Plot Accuracy Before and After SMOTE
  ```ruby
  # Get classification reports before and after SMOTE
  classification_report_before_smote = classification_report(y_test, y_pred_before_smote, output_dict=True)
  classification_report_after_smote = classification_report(y_test, y_pred_after_smote, output_dict=True)

  # Extract accuracy values
  accuracy_before_smote = classification_report_before_smote['accuracy']
  accuracy_after_smote = classification_report_after_smote['accuracy']

  # Plot accuracy before and after SMOTE
  labels = ['Before SMOTE', 'After SMOTE']
  accuracy_scores = [accuracy_before_smote, accuracy_after_smote]

  plt.bar(labels, accuracy_scores, color=['blue', 'green'])
  plt.xlabel('SMOTE')
  plt.ylabel('Accuracy')
  plt.title('Accuracy of Logistic Regression before and after SMOTE')
  plt.ylim(0, 1)  # Limit y-axis from 0 to 1 for accuracy range
  plt.show()
  ```
  ![LogReg_Accuracy_Before and after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/63740f0e-86f4-433d-bc0c-3cb69da4c3bf)

  > Advantages: <br>
  >> - Logistic Regression performs well when the dataset is linearly separable <br>
  >> - Logistic Regression not only gives a measure of how relevant a predictor (coefficient size) is, but also its direction of association (positive or negative) <br>
  
  > Disadvantages: <br>
  >> - Main limitation of Logistic Regression is the assumption of linearity between the   dependent variable and the independent variables <br>
  >> - If the number of observations are lesser than the number of features, Logistic Regression should not be used, otherwise it may lead to overfit  <br>



- [Neural Network Model](https://subscription.packtpub.com/book/data/9781788397872/1/ch01lvl1sec27/pros-and-cons-of-neural-networks): <br>
  Step 1. Import library <br>
  ```ruby
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense
  ```
  Step 2. Scale features (optional but often recommended)
  ```ruby
  scaler = StandardScaler()
  X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
  X_test_scaled = scaler.transform(X_test)
  ```
  Step 3. Define the neural network model architecture before SMOTE <br>
  ```ruby
  lmodel_before_smote = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
  ])
  ```
  Step 4. Compile the model <br>
  ```ruby
  model_before_smote.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  ```
  Step 5. Train the model before SMOTE
  ```ruby
  model_before_smote.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
  ```
  Step 6. Make predictions on the test set before SMOTE <br>
  ```ruby
  y_pred_before_smote = (model_before_smote.predict(X_test) > 0.5).astype("int32")
  ```
  Step 7. Compute and print confusion matrix and classification report before SMOTE
  ```ruby
  print("Confusion Matrix and Classification Report before SMOTE:")
  print(confusion_matrix(y_test, y_pred_before_smote))
  print(classification_report(y_test, y_pred_before_smote))
  ```
  ![NN_Before Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/b5dbe9b3-47b7-438a-bd50-7aa594a65a3e)

  Step 8. Define the neural network model architecture after SMOTE
  ```ruby
  model_after_smote = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_resampled_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
  ])
  ```

  Step 9. Compile the model
  ```ruby
  model_after_smote.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])
  ```

  Step 10. Train the model after SMOTE
  ```ruby
  history_smote = model_after_smote.fit(X_train_resampled_scaled, y_train_resampled, epochs=10, batch_size=32, verbose=1)
  ```

  Step 11. Make predictions on the test set after SMOTE
  ```ruby
  y_pred_after_smote = (model_after_smote.predict(X_test_scaled) > 0.5).astype("int32")
  ```

  Step 12. Compute and print confusion matrix and classification report after SMOTE
  ```ruby
  print("Confusion Matrix and Classification Report after SMOTE:")
  print(confusion_matrix(y_test, y_pred_after_smote))
  print(classification_report(y_test, y_pred_after_smote))
  ```

  ![NN After Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/5dbf196c-9751-43e7-a87c-e7541b3d0b12)


  Step 13. Plot confusion matrix before and after SMOTE
  ```ruby
  # Plot confusion matrix before SMOTE
  axes[0].imshow(confusion_matrix_before_smote, cmap=plt.cm.Blues, interpolation='nearest')
  axes[0].set_title('Confusion Matrix Before SMOTE')
  axes[0].set_xticks([0, 1])
  axes[0].set_yticks([0, 1])
  axes[0].set_xlabel('Predicted Label')
  axes[0].set_ylabel('True Label')
  for i in range(2):
      for j in range(2):
          axes[0].text(j, i, str(confusion_matrix_before_smote[i, j]),
                       horizontalalignment='center', verticalalignment='center', color='white')

  # Plot confusion matrix after SMOTE
  axes[1].imshow(confusion_matrix_after_smote, cmap=plt.cm.Blues, interpolation='nearest')
  axes[1].set_title('Confusion Matrix After SMOTE')
  axes[1].set_xticks([0, 1])
  axes[1].set_yticks([0, 1])
  axes[1].set_xlabel('Predicted Label')
  axes[1].set_ylabel('True Label')
  for i in range(2):
      for j in range(2):
          axes[1].text(j, i, str(confusion_matrix_after_smote[i, j]),
                       horizontalalignment='center', verticalalignment='center', color='white')

  plt.tight_layout()
  plt.show()
  ```

  ![NN_ConfusionMetrics](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/436f2328-888d-4abd-8ed2-397e03db62ca)


  Step 14. Plot Accuracy Before and After SMOTE
  ```ruby
  # Plot accuracy before and after SMOTE
  plt.plot(history_original.history['accuracy'], label='Original Dataset')
  plt.plot(history_smote.history['accuracy'], label='After SMOTE')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.title('Accuracy of Neural Network before and after SMOTE')
  plt.legend()
  plt.show()
  ```
  | Plot Accuracy 1 | Plot Accuracy 2 | 
  | :---: | :---: | 
  | ![NN_Accuracy Before after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/d88fa925-694f-47bf-a0f3-5edd86021790) | ![NN_Diagram Accuracy before after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/d7b4fb46-431c-46a7-9e85-fcd8128b0999) | 

  > Advantages: <br>
  >> - Neural Network can be trained with any number of inputs and layers <br>
  >> - Neural networks work best with more data points
  >> - One trained, the predictions are pretty fast
  
  >  Disadvantages: <br>
  >> - Neural networks are black boxes, meaning we cannot know how each independent variable is influencing the dependent variables <br>
  >> - It is computationally very expensive and time consuming to train with traditional CPUs  <br>
  >> - Neural networks depend a lot on training data





- [Support Vector Machine Model:](https://scikit-learn.org/stable/modules/svm.html) <br>
  Step 1. Import library <br>
  ```ruby
  from sklearn.svm import SVC
  ```
  Step 2. Initialize SVM classifier before SMOTE
  ```ruby
  svm_classifier_before_smote = SVC(kernel='rbf', random_state=42)
  ```
  Step 3. Train SVM classifier before SMOTE <br>
  ```ruby
  svm_classifier_before_smote.fit(X_train, y_train)
  ```
  Step 4. Make predictions on the test set before SMOTE <br>
  ```ruby
  y_pred_before_smote = svm_classifier_before_smote.predict(X_test)
  ```
  Step 5. Compute and print confusion matrix and classification report before SMOTE
  ```ruby
  print("Confusion Matrix and Classification Report before SMOTE:")
  print(confusion_matrix(y_test, y_pred_before_smote))
  print(classification_report(y_test, y_pred_before_smote))
  ```

  ![SVM_Before smote Confusion](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/5ca15720-e31d-4c6c-a0be-6e59297cb226)


  Step 6. Train SVM classifier after SMOTE <br>
  ```ruby
  svm_classifier_after_smote.fit(X_train_resampled_scaled, y_train_resampled)
  ```
  Step 7. Make predictions on the test set after SMOTE
  ```ruby
  y_pred_after_smote = svm_classifier_after_smote.predict(X_test_scaled)
  ```
  Step 8. Compute and print confusion matrix and classification report after SMOTE
  ```ruby
  print("Confusion Matrix and Classification Report after SMOTE:")
  print(confusion_matrix(y_test, y_pred_after_smote))
  print(classification_report(y_test, y_pred_after_smote))
  ```
  
  ![SVM_After SMOTE Confusion](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/071e71d2-8392-41c2-9f1a-0f4938a903e1)

  Step 9. Plot Confusion Metrics Before and After SMOTE
  ```ruby
  # Plot confusion matrices
  fig, axes = plt.subplots(1, 2, figsize=(12, 6))

  # Plot confusion matrix before SMOTE
  axes[0].imshow(confusion_matrix_before_smote, cmap=plt.cm.Blues, interpolation='nearest')
  axes[0].set_title('Confusion Matrix Before SMOTE')
  axes[0].set_xticks([0, 1])
  axes[0].set_yticks([0, 1])
  axes[0].set_xlabel('Predicted Label')
  axes[0].set_ylabel('True Label')
  for i in range(2):
      for j in range(2):
          axes[0].text(j, i, str(confusion_matrix_before_smote[i, j]),
                       horizontalalignment='center', verticalalignment='center', color='white')

  # Plot confusion matrix after SMOTE
  axes[1].imshow(confusion_matrix_after_smote, cmap=plt.cm.Blues, interpolation='nearest')
  axes[1].set_title('Confusion Matrix After SMOTE')
  axes[1].set_xticks([0, 1])
  axes[1].set_yticks([0, 1])
  axes[1].set_xlabel('Predicted Label')
  axes[1].set_ylabel('True Label')
  for i in range(2):
      for j in range(2):
          axes[1].text(j, i, str(confusion_matrix_after_smote[i, j]),
                       horizontalalignment='center', verticalalignment='center', color='white')

  plt.tight_layout()
  plt.show()
  ```

  ![Plot confusion metric SVM](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/e5a86814-f223-45ea-892e-22bda98e9af1)


  Step 10. Plot Accuracy Before and After SMOTE
  ```ruby
  # Get classification reports before and after SMOTE
  classification_report_before_smote = classification_report(y_test, y_pred_before_smote,   output_dict=True)
  classification_report_after_smote = classification_report(y_test, y_pred_after_smote, output_dict=True)

  # Extract accuracy values
  accuracy_before_smote = classification_report_before_smote['accuracy']
  accuracy_after_smote = classification_report_after_smote['accuracy']

  # Plot accuracy before and after SMOTE
  labels = ['Before SMOTE', 'After SMOTE']
  accuracy_scores = [accuracy_before_smote, accuracy_after_smote]

  plt.bar(labels, accuracy_scores, color=['blue', 'green'])
  plt.xlabel('SMOTE')
  plt.ylabel('Accuracy')
  plt.title('Accuracy of SVM before and after SMOTE')
  plt.ylim(0, 1)  # Limit y-axis from 0 to 1 for accuracy range
  plt.show()
  ```
  ![SVM Accuracy Before adn After SVM](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/0cee870f-92b6-4a87-87f0-d3ada2d971e9)

  > Advantages: <br>
  >> - Effective in High Dimensional Spaces <br>
  >> - Still effective in cases where number of dimensions is greater than the number of samples
  >> - Uses a subset of training points in the decision function/support vectors, so it is also memory efficient
  >> - Versatile: different kernel functions can be specified for the decision function.

  > Disadvantages: <br>
  >> - If the number of features is much greater than the number of samples, avoid over-fitting in choosing kernel functions and regularization term is crucial <br>
  >> - SVM do not directly provide probability estimates, these are calculated using an expensive five fold cross validation<br>

- Which algorithm is best for prediction? <br>
  | Logistic Regression | Neural Network | Support Vector Machine |
  | :---: | :---: | :---: | 
  | ![LogReg_Accuracy_Before and after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/051a8bc0-2c44-4e54-a377-18e6c0e9e5bb) | ![NN_Diagram Accuracy before after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/635bc08d-d78f-4b57-b6c2-c71549879242) | ![SVM Accuracy Before adn After SVM](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/ac82d963-2026-4273-bcdc-2ed9f84a5033) | <br>

  The best algorithms for predicion Binary Integer Data in this case is ***Neural Network + SMOTE*** with accuracy 98%. Based on the plotting of accuracy results between NN Before and After SMOTE, Neural Network can automatically learn relevant features from raw data and have a flexibility in various architectures which enables them to capture different types of patterns and structures in the data, making them adaptable to diverse classification tasks across various domains. In this model, impelenting Activation RelU Function in input layer, hidden layer and Acitvation Sigmoid Function in Output layer are the main reason why Neural Network is the most appropriate algorithms for classifying Target Labelled into binary ineteger data. <br>

   ![NN_Accuracy Before after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/6c8ead2e-918c-4eb4-ba1f-c364b1270279)

## Evaluation
- What metrics evaluation is used in this project and how it works? <br> 
The metrics evaluation used are [Confusion Metrics, Accuracy, Precision, Sensitivity(Recall), Specificity and F1-Score](https://medium.com/javarevisited/evaluating-the-logistic-regression-ae2decf42d61). Those performance metrics are commonly used to evaluate a Binary Classification Model. Let's define each metric and how they work below: <br>
1. Confusion Metric: <br>
   A Tabular summary of True/False and Positive/Negative prediction rates. It allows to compute various performance metrics. <br>

   ![Basic-Confusion-matrix](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/2fd76ef0-c24e-4ba3-ae8a-0959bf5b1e9a)
   
2. Accuracy: <br>
measures the ratio of correct predictions from all predicted results. <br>

   ![accurcay score measuring](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/dd2b037c-cb5d-4cbd-a49c-547d67154a16)

3. Precision: <br>
measures what proportion of the positive predictions is actually positive. The precision score is useful for the succes of prediction when classes are very imbalanced and when it's significantly cost-efficient to identify all positive examples without any false positive. <br>

   ![precision formula](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/8d654914-c0eb-4047-95f6-e72aa5e9efbb)

4. Sensitivity/Recall: <br>
represents the model’s ability to correctly predict the positives out of actual positives. The higher the recall score, the better the machine learning model is at identifying positives. <br>
   ![recall formula](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/8a47bebf-2e71-4b39-8147-8d90b0ff3673)

5. Specificity: <br>
represents the model’s ability to correctly predict the negatives out of actual negatives. The higher the specificity score, the better the machine learning model is at identifying negatives. <br>

   ![specificity](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/0a7141f5-3bf6-4c89-bfcf-51010985190a)

6. F1 Score: <br>
combine the precision and recall of the model, and it is the harmonic mean of the precision and recall. It’s used often when data is imbalanced. <br>

   ![f1 score](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/4a047010-c742-47c0-a375-015bf2b223c8)

 
- Explanation of The Metrics Evaluation Project Results! <br>
  | X | Logistic Regression | Neural Network |  Support Vector Machine | 
  | :---: | :---: | :---: | :---: | 
  | Confusion Metrics Before SMOTE | ![LogReg_Before Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/4d073946-e2b4-410a-8bb5-749e25f9e604) | ![NN_Before Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/ffcb28c5-012c-4f6e-a493-563209fc2b98)| ![SVM_Before smote Confusion](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/29275888-f04f-41a2-a6a8-1a9d46e53f59) | 
  | Confusion Metrics After SMOTE | ![LogReg_After Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/8321ccc2-2688-4992-8264-727ddec9b2ea) |![NN After Smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/ddc13079-5354-42ee-977f-c813b8331355) | ![SVM_After SMOTE Confusion](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/e5d4a8a8-b87f-4782-8a2c-f8091f30e178)|
  | Plot Accuracy |  ![LogReg_Accuracy_Before and after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/051a8bc0-2c44-4e54-a377-18e6c0e9e5bb) | ![NN_Diagram Accuracy before after smote](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/635bc08d-d78f-4b57-b6c2-c71549879242) | ![SVM Accuracy Before adn After SVM](https://github.com/adinplb/Belajar-Machine-Learning-Terapan-Dicoding/assets/61041719/ac82d963-2026-4273-bcdc-2ed9f84a5033) |

  To racap, on the comparison of confusion matrix and accuracy values plot results above, Neural Network with SMOTE has the highest accuracy score and great confusion matrics so this algorithms is the most suitable and approriate model for predictiong Breast Tumor Diagnosis using Quantitative Cell-Nuclear Phenotypes Features. The flexibility of Neural Network Architectures makes enables to capture different types of patterns/structures in dataset and learn relevant numerical features to predict Binary Integer Data whether it is Benign (0) or Malignant (1).

## Conclusion 
1. Not all features have an impact to the algorithms in the model prediction.
2. Neural Network with handling imbalanced data using SMOTE, give the greatest accuracy so this algorithm is the best model for predicting Breast Tumor Diagnosis.
3. Yes, SMOTE influences the high accuracy of the model.
