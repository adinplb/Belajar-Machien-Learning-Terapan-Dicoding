# First Project Report of Applied ML
>### ***"Breast Tumor Prediction and Diagnosis Using Quantitative Cell Nuclear Phenotype Features in Supervised Machine Learning Algorithms"***
>>#### Issued by **Muhammad Adin Palimbani**

<img src="https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/blob/02935c3aedf355fbce58e95d36dfaf547f90fab5/images/Histology-of-left-breast-cancer-The-tumor-is-composed-of-large-nests-with-central-comedo.png" width="500"/> <img src="https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/de1f9259-51cd-4e2f-afe1-9a1e9e55dbdf" width="330"/> 

## Project Domain 
1. Issue Focus? <br>
The significant advances in cancer research over the past decades has been carried out with the advent of new technologies in the field of medicine. Scientists have conducted a new approach with different methods for the early prediction of cancer treatment outcome particularly Breast Cancer. One of the example approaches applied is the growing trend on Machine Learning Techniques. However, a common problem in several research is the lack of external validation or testing regarding the predictive performance of their models and also imbalanced data handling.  This may lead to malformed prediction models and system failures at the production stage

2. Why does the issue need to be resolved? <br>
The accurate prediction models of a disease outcome is extremely depends on the medical data of the patient. Medical data contains the patient's details condition and diagnosis which hold unnecessary and interrelated data. Those data is high dimensional data as well in particular the integration of clinical and genomic mixed data. In several studies, scientists have proved that approaches related to the genomic characteristics provides promising results for cancer detection and identification, for instances, digitized image of a fine needle aspirate (FNA) of a breast mass which represent cell nuclear characteristics in Breast Tumor. However, these methods suffer from low sensitivity regarding their use in screening at early stages and difficulty to determine benign from malignant tumors. This is the reason why the cancer predictive performance models issue need to be resolved in order to prevent malformed prediction and system failures. 

3. How to address the issue? <br>
Interactive image processing techniques, along with a linear programming based inductive classifier, have been used to creeate a highly accurate system for diagnosis of Breast Tumors. A small fraction of a Fine Needle Aspirate Slide (FNA) is selected and digitized. The digitized image of a FNA of a Breast mass describe chracteristics of the cell nuclei present in the image. Those are computed and become features for this research. So we could possibily diagnosed whether it is Malignant or Benign through Nuclear Feature Extraction.

4. Related References from Credible Sources: <br> 
[Nuclear Feature Extraction For Breast Tumor Diagnosis](https://minds.wisconsin.edu/bitstream/handle/1793/59692/TR1131.pdf;jsessionid=0449D8C1D78CAAB2BF57B76AABE87312?sequence=1). <br>
[Prediction of parameters of liver tumor using feature extraction and supervised function](https://www.sciencedirect.com/science/article/pii/S2665917422000204). <br>
[Machine Learning Algorithms For Breast Cancer Prediction And Diagnosis](https://www.sciencedirect.com/science/article/pii/S1877050921014629). <br>
[Quantitative nuclear phenotype signatures predict nodal disease in oral squamous cell carcinoma](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8568158/). <br>

## Business Understanding
Breast Tumor Diagnosis has been conducted by [Fine Needle Aspiration (FNA)](https://cancer.ca/en/treatments/tests-and-procedures/fine-needle-aspiration-fna); a type of biopsy which uses a very thin needle and syringe to remove a sample of cells, tissue or fluid from an abnormal area or lump in the body. FNAs has been able to diagnose successfully in examining the cell nuclear phenotypes and become a features which indicates a higher likelihood of malignancy. The computer vision diagnostics system extracts 10 different features from the snake-generated cell nuclei boundaries. Those extracted features are numerically modeled which consist of ***Radius***, ***Perimeter***, ***Area***, ***Compactness***, ***Smoothness***, ***concavity***, ***Concave Points***, ***Symmetry***, ***Fratal Dimension*** and ***Texture***. In addition, there is a diagnosis breast tumor features which represent a malignant or bening so in this project there is a target labelled to predict whether it is a benign or malignant tumor. A Supervised learning model is suitable for this problem by using the quantitative cell nuclear phenotype of Breast Tumor and followed by handling imbalanced data.

### Problem Statement
1. Does each feature in this dataset have an influence on breast tumor prediction?
2. Which Machine Learning model can solve the problem and present the best model as a solution?
3. How does SMOTE affect model predictions?

### Goals
1. Find features that have an influence on breast tumor prediction
2. Find the best Machine Learning Model that could solve the problem
3. Find the affect of SMOTE in handling imbalanced data for model prediction

### Solution Statements
To reach out good Breast Cancer Prediction, using 4 different type of Regression model for binary integer data in Supervised Machine Learning Algorithms. These are suitable for predicting the target labelled where the output are 0 (Benign) and 1 (Malignant). In addtion, SMOTE for handling imbalanced data will be implement as well. The algorithms are as follows: <br>

- [Logistic Regression](https://nthu-datalab.github.io/ml/labs/06_Logistic-Regression_Metrics/06_Logistic-Regression_Metrics.html) <br>
Logistic Regression is a classification algorithm in combination with a decision rule that makes dichotomous the predicted probabilities of the outcome. Currently, it is one of the most widely used classification models in Machine Learning. So far, we evaluate the performance of a classifier using the accuracy metric. Although accuracy is a general and common metric, there are several other evaluation metrics that allow us to quantify the performance of a model from different aspects.
- [Neural Network](https://www.analyticsvidhya.com/blog/2021/11/neural-network-for-regression-with-tensorflow/) <br>
TensorFlow can be used for regression tasks. It provides a flexible platform to build and train neural networks for regression problems. MAE is a very simple metric which calculates the absolute difference between actual and predicted values
- [Support Vector Machine](https://medium.com/@mkk.rakesh/support-vector-machine-explained-with-a-binary-classification-problem-bb1d5be336c4) <br>
SVM (Support Vector Machines) distinguishes itself from other Machine Learning models with its proficiency in handling high-dimensional data, finding complex decision boundaries. SVM offers various kernels for both linear and non-linear data, and maintaining robust performance even when data is limited. To evaluate SVM models for classification tasks, it's more appropriate to use classification-specific metrics such as accuracy, precision, recall, F1-score, and area under the ROC curve (ROC-AUC). These metrics provide insights into the SVM model's ability to classify instances correctly across different classes and account for the inherent characteristics of classification tasks.
- [Random Forest](https://www.ibm.com/topics/random-forest) <br>
Random forest is a commonly-used machine learning algorithm trademarked by Leo Breiman and Adele Cutler, which combines the output of multiple decision trees to reach a single result. Its ease of use and flexibility have fueled its adoption, as it handles both classification and regression problems. Metrics, such as Gini impurity, information gain, or [mean square error (MSE)](https://www.ibm.com/topics/random-forest#:~:text=Metrics%2C%20such%20as%20Gini%20impurity,%22don't%20surf.%22), can be used to evaluate the quality of the split. This decision tree is an example of a classification problem, where the class labels are "surf" and "don't surf."


## Data Understanding
Dataset used are collected from Kaggle. It is [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/code). Dataset is in CSV File Format which total class contribution are 357 Benign and 212 Malignant. Dataset is consist of 569 rows and 33 Features; 1 Categorical Features and 32 Numerical Features. One Categorical Features will be converted into Integer as Target Labelled. Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. All feature values are recorded with four significant digits and none missing values. The extracted features are as follows: <br>
1. ***Radius***: The radius of and individual nucleaus is measured by averaging the length of the radial line segments defined by the centeroid of the snake and the individual snake points.
2. ***Perimeter***: The total distance between the snake point constitues the nuclear perimeter.
3. ***Area***: Nuclear area is measured simply by counting the number of pixels on the interior of the snake and adding one-hald of the pixels in the perimeter.
4. ***Compactness***: Perimeter and area are combined to give a measure of the compactness of the cell nuclei using the formula perimeter<sup>2</sup>/area.
5. ***Smoothness***: Quantified by measuring the difference between the length of a radial line and the mean length of the lines surrounding it.
6. ***Concavity***: Severity of concave portions of the contour
7. ***Concave points***: Number of concave portions of the contour
8. ***Symmetry***: In order to measure symmetry, the major axis, or longest chord through the center, is found.
9. ***Fractal Dimension***: Approximated using the "Coastline Approximation" described by Mandelbrot.
10. ***Texture***: meaasured by finding the variance of the gray scale intensities in the component pixels.<br>
11. ***Diagnostic***: Benign and Malignant

>_mean, _se (standar error) and _worst (largest): mean of the 3 largest values of these features were computed for each image, resulting in 30 features. For instance: field 3 is mean radius, field 13 is radius_se, field 23 is worst radius.

 . | radius_mean | radius_se | radius_worst | 
 --- | --- | --- | --- | 
Definition | mean of distances from center to points on the perimeter | standard error for the mean of distances from center to points on the perimeter | "worst" or largest mean value for mean of distances from center to points on the perimeter | 
Values | 17.99 | 1.095| 25.38 | 

### Exploratory Data Analysis and Visualization
#### Check Data Type
```ruby
df.info()
```
![df info](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/34570f7b-7e16-414e-89f6-0599f85d0f78)

#### Check Missing Values
```ruby
print(df.isna().sum())
```
![df isna](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/e14838d5-5672-4119-95dd-fd14397d81d0)

#### Check Data Duplication
```ruby
print("Jumlah yang terduplikasi:", df.duplicated().sum())
```
![df duplicated](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/78819fb9-3ed0-47f7-97fa-da9328168a27)

#### Descriptive Statistics Analysis
```ruby
df.describe()
```
![df describe](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/9b0f5c49-5927-4b31-adad-21dcd35e6f67)

#### Check Outliers
Four Example Features used to Check Outliers:
```ruby
sns.boxplot(x=df['radius_mean'])
```

![radius mean](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/319ae8e9-47d5-46df-b475-69291e915794)

```ruby
sns.boxplot(x=df['texture_mean'])
```
![texture mean](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/2b529395-c8a8-4c5d-8dd4-68ab9de92e6a)

```ruby
sns.boxplot(x=df['perimeter_mean'])
```
![perimeter mean](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/e3edbf37-30f9-4cca-88b8-06df171c8002)

```ruby
sns.boxplot(x=df['area_mean'])
```
![area mean](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/cac1e7b7-c579-4d9a-b860-67c326c7a144)

#### Univariate Analysis
```ruby
df.hist(bins=50, figsize=(20,15))
plt.show()
```
![univariate analysis](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/ac1b6f89-6e0e-4a0f-a564-19ed276c37bc)

#### Multivariate Analysis
```ruby
# Observe relation among numerical features using pairplot() 
sns.pairplot(df, diag_kind = 'kde')
```
![pairplot](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/0b1edff5-93de-4089-8fdc-1fe13e92836f)

```ruby
plt.figure(figsize=(20, 18))
correlation_matrix = df.corr().round(2)
# parameter anot=true is used for printing value inside the box
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths= 0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)
```
![multivariate](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/06078f33-9f1a-4f04-849f-90fb6d9e97de)

## Data Preparation
At this stage, PCA feaures reduction, change target labelled type into binary integer, IQR Method, SMOTE and Feature Scalling are approriate techniques for this type of dataset. Moreover, the class contribution in dataset are indeed imbalanced; 357 Benign and 212 Malignant so SMOTE or Synthetic Minority Over-sampling Technique will be implemented. Removing outliers will be performed as well and followed by feature scaling or z-score normalization where they have a mean of 0 and a standard deviation of 1. The data size will be splitted into train set and test set with ratio 80:20. To understand deeply the ins and outs of data preparation is by looking at these several steps: <br>

1. Convert "Diagnosis" Feature "object" type into "binary integer" values 0 and 1.
>Why is it necessary for this technique to be carried out?
>>Answer: require input features to be numerical so the algorithms can be performed.
```ruby
df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
```
<img src="https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/a29b20da-b0f5-4b34-b066-2c2ed1e285d9"/> <img src="https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/89166fd5-dce2-499c-8a6d-3909cafa9db2"/> 

2. Remove outliers using IQR Method in all Features. Then, check data shape.
>Why is it necessary for this IQR Method to be carried out?
>>Answer: outliers can increase variance in the dataset and the Interquartile Range (IQR) method is a robust and commonly used technique for detecting and removing outliers as well. 
```ruby
Q1 = df_baru.quantile(0.25)
Q3 = df_baru.quantile(0.75)
IQR=Q3-Q1
df_baru=df_baru[~((df_baru<(Q1-1.5*IQR))|(df_baru>(Q3+1.5*IQR))).any(axis=1)]

# Check data shape after dropping outliers
df_baru.shape
```
![data shape after drop outliers](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/26534822-5f9f-451a-9efe-cbccc9b7c3eb) <br>
The dataset has been cleaned and has 398 samples

3. Reduce dimension of radius_mean, perimeter_mean, area_mean, radius_worst, perimeter_worst and area_worst feature using PCA
>Why is it necessary for reducing those features using PCA to be carried out?
>>Answer: The pairplot result shows those six features have a high correlation so they can be reduced using PCA which help to reduce noise and redundancy in dataset.
```ruby
from sklearn.decomposition import PCA
pca = PCA(n_components=1, random_state=123)
pca.fit(df_baru[['radius_mean', 'perimeter_mean', 'area_mean', 'radius_worst', 'perimeter_worst', 'area_worst']])
df_baru['dimension'] = pca.transform(df_baru.loc[:, ('radius_mean', 'perimeter_mean', 'area_mean', 'radius_worst', 'perimeter_worst', 'area_worst')]).flatten()
df_baru.drop(['radius_mean', 'perimeter_mean', 'area_mean', 'radius_worst', 'perimeter_worst', 'area_worst'], axis=1, inplace=True)
df_baru
```
![dimension](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/bf28fcf9-1c18-4ecd-8284-fd15c807384a)

4. Splitting Data into Train Set and Test Set + SMOTE For imbalanced data
>Why is it necessary for handling imbalanced data using SMOTE to be carried out?
>>Answer: To maximize overall accuracy and minimize MSE which can be misleading when classes are imbalanced and SMOTE (Synthetic Minority Over-sampling Technique) is one of the method used to address this issue.
```ruby
pip install imbalanced-learn
```
```ruby
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Assuming X contains your features and y contains the corresponding labels
# Perform train-test split

X = df.drop(["diagnosis"],axis =1)
y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE only on the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Now, X_train_resampled and y_train_resampled contain the resampled data using SMOTE,
# while X_test and y_test remain unchanged

# Proceed with model training and evaluation using the resampled training data and the original test data

# Count the number of samples in each class before and after SMOTE
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_train_resampled, counts_train_resampled = np.unique(y_train_resampled, return_counts=True)

# Create a DataFrame to display the results
data = {
    'Class': unique_train,
    'Original Count': counts_train,
    'Resampled Count': counts_train_resampled
}

df_result = pd.DataFrame(data)
print("Before SMOTE:")
print(df_result)

# Visualize the distribution of classes after SMOTE
print("\nAfter SMOTE:")
print("Classes:", unique_train_resampled)
print("Counts:", counts_train_resampled)
```
![output smote](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/0c0b9474-ad9f-4f9a-9482-8648a8642ae4)

5. Feature Scalling using Z-Score Normalization
>Why is it necessary for feature scalling to be carried out?
>>Answer: To ensures all features have a similar scale, typically between 0 and 1 or around a mean of 0 with a standard deviation of 1. 
```ruby
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()
X_train[numerical_features].describe().round(4)
```
![standarisation](https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/ed2b1602-7ea1-4089-a6ed-c9d7c9037585)


## Modelling
- Logistic Regression Model: <br>
  Step 1. <br>
  Step 2. <br>
  Step 3. <br>
  > Advantages: <br>
  >> - Lorem Ipsum <br>
  >> - Lorem Ipsum 
  
  > Disadvantages: <br>
  >> - Lorem Ipsum <br>
  >> - Lorem Ipsum  <br>

- Neural Network Model: <br>
  Step 1. <br>
  Step 2. <br>
  Step 3. <br>
  > Advantages: <br>
  >> - Lorem Ipsum <br>
  >> - Lorem Ipsum 
  
  >  Disadvantages: <br>
  >> - Lorem Ipsum <br>
  >> - Lorem Ipsum  <br>

- Support Vector Machine Model: <br>
  Step 1. <br>
  Step 2. <br>
  Step 3. <br>
  > Advantages: <br>
  >> - Lorem Ipsum <br>
  >> - Lorem Ipsum
  
  > Disadvantages: <br>
  >> - Lorem Ipsum <br>
  >> - Lorem Ipsum  <br>

- Random Forest Model: <br>
  Step 1. <br>
  Step 2. <br>
  Step 3. <br>
  > Advantages: <br>
  >> - Lorem Ipsum <br>
  >> - Lorem Ipsum
  
  > Disadvantages: <br>
  >> - Lorem Ipsum <br>
  >> - Lorem Ipsum  <br>

- Which the best algorithms for prediction and why? 

## Evaluation
- What metric evalution used in this case?
- Exaplain project result based on metric evaluation
- Menjelaskan metrik evaluasi yang digunakan untuk mengukur kinerja model. Misalnya, menjelaskan formula metrik dan bagaimana metrik tersebut bekerja

## Conclusion 
1. Tidak semua fitur memiliki pengaruh dalam prediksi model
2. Neural Network dengan penangana imbalanece data memberikan nilai MSE paling kecil sehingga algoritma ini menjadi model terbaik
3. Iya, SMOTE memberikan tingkat akurasi yang tinggi dan nilai MSE yang rendah
