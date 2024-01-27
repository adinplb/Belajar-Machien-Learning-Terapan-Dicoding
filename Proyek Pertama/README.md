# First Project Report of Applied ML
>### ***"Breast Tumor Prediction and Diagnosis Using Quantitative Cell Nuclear Phenotype Features in Supervised Machine Learning Algorithms"***
>>#### Issued by **Muhammad Adin Palimbani**

<img src="https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/blob/02935c3aedf355fbce58e95d36dfaf547f90fab5/images/Histology-of-left-breast-cancer-The-tumor-is-composed-of-large-nests-with-central-comedo.png" width="500"/> <img src="https://github.com/adinplb/Belajar-Machien-Learning-Terapan-Dicoding/assets/61041719/de1f9259-51cd-4e2f-afe1-9a1e9e55dbdf" width="330"/> 

## Project Domain 
1. Issue Focus? <br>
The significant advances in cancer research over the past decades has been carried out with the advent of new technologies in the field of medicine. Scientists have conducted a new approach with different methods for the early prediction of cancer treatment outcome particularly Breast Cancer. One of the example approaches applied is the growing trend on Machine Learning Techniques. However, a common problem in several research is the lack of external validation or testing regarding the predictive performance of their models. This may lead to malformed prediction models and system failures at the production stage

2. Why does the issue need to be resolved? <br>
The accurate prediction models of a disease outcome is extremely depends on the medical data of the patient. Medical data contains the patient's details condition and diagnosis which hold unnecessary and interrelated data. Those data is high dimensional data as well in particular the integration of clinical and genomic mixed data. In several studies, scientists have proved that approaches related to the genomic characteristics provides promising results for cancer detection and identification, for instances, digitized image of a fine needle aspirate (FNA) of a breast mass which represent cell nuclear characteristics in Breast Tumor. However, these methods suffer from low sensitivity regarding their use in screening at early stages and difficulty to determine benign from malignant tumors. This is the reason why the cancer predictive performance models issue need to be resolved in order to prevent malformed prediction and system failures. 

3. How to address the issue? <br>
Interactive image processing techniques, along with a linear programming based inductive classifier, have been used to creeate a highly accurate systm for diagnosis of Breast Tumors. A small fraction of a Fine Needle Aspirate Slide (FNA) is selected and digitized. The digitized image of a FNA of a Breast mass describe chracteristics of the cell nuclei present in the image. Those are computed and become features for this research. So we could possibily diagnosed whether it is Malignet or Benign through Nuclear Feature Extraction.

4. Related References from Credible Sources: <br> 
[Nuclear Feature Extraction For Breast Tumor Diagnosis](https://minds.wisconsin.edu/bitstream/handle/1793/59692/TR1131.pdf;jsessionid=0449D8C1D78CAAB2BF57B76AABE87312?sequence=1). <br>
[Prediction of parameters of liver tumor using feature extraction and supervised function](https://www.sciencedirect.com/science/article/pii/S2665917422000204). <br>
[Machine Learning Algorithms For Breast Cancer Prediction And Diagnosis](https://www.sciencedirect.com/science/article/pii/S1877050921014629). <br>
[Quantitative nuclear phenotype signatures predict nodal disease in oral squamous cell carcinoma](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8568158/). <br>

## Business Understanding
Breast Tumor Diagnosis has been conducted by [Fine Needle Aspiration (FNA)](https://cancer.ca/en/treatments/tests-and-procedures/fine-needle-aspiration-fna); a type of biopsy which uses a very thin needle and syringe to remove a sample of cells, tissue or fluid from an abnormal area or lump in the body. FNAs has been able to diagnose successfully in examining the cell nuclear phenotypes and become a features which indicates a higher likelihood of malignancy. The computer vision diagnostics system extracts 10 different features from the snake-generated cell nuclei boundaries. Those extracted features are numerically modeled which consist of ***Radius***, ***Perimeter***, ***Area***, ***Compactness***, ***Smoothness***, ***concavity***, ***Concave Points***, ***Symmetry***, ***Fratal Dimension*** and ***Texture***. In addition, there is a diagnosis breast tumor features which represent a malignant or benig so in this project there is target labelled to predict whether it is a benign or malignant tumor. A Supervised learning model is suitable for this problem by using the quantitative cell nuclear phenotype of Breast Tumor.

### Problem Statement
1. Does each feature in this dataset have an influence on breast tumor prediction?
2. Which Machine Learning model can solve the problem and present the best model as a solution?

### Goals
1. Find features that have an influence on breast tumor prediction
2. Find the best Machine Learning Model that could solve the problem

### Solution Statements
To reach out good Breast Cancer Prediction, using 3 different type binary classification model in Supervised Machine Learning Algorithms. These are suitable for predicting the target labelled where the output are 0 (Benign) and 1 (Malignant). The algorithms are as follows: 
- Linear Regression
  Harus terukur dg metrik evaluasi 
- K Nearest Neigbors
   Harus terukur dg metrik evaluasi 
- Random Forest
   Harus terukur dg metrik evaluasi 
- Support Vector Machine
   Harus terukur dg metrik evaluasi 

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
At this stage, PCA and One-Hot Encoding are approriate techniques for features reduction and represent category-type data into binary integer values 0 and 1. Moreover, the class contribution in dataset are indeed imbalanced; 357 Benign and 212 Malignant so SMOTE or Synthetic Minority Over-sampling Technique will be implemented. Removing outliers will be performed as well and followed by feature scaling or z-score normalization where they have a mean of 0 and a standard deviation of 1. The data size will be splitted into train set and test set with ratio 80:20. To understand deeply the ins and outs of data preparation is by looking at these several steps: <br>

1. Convert "Diagnosis" Feature "object" type into "binary integer" values 0 and 1 using One Hot Encoding.
>Why One Hot Encoding?
```ruby
code in notebook
```

2. Remove outliers using IQR Method in all Features. Then, check data shape.
>Why this is neccessary do?
```ruby
code in notebook
```

3. Reduce dimension of radius_mean, perimeter_mean, area_mean, radius_worst, perimeter_worst and area_worst feature using PCA
>Why this is neccessary do??
```ruby
code in notebook
```

4. SMOTE For imbalance data
>Why this is neccessary do??
```ruby
code in notebook
```

5. Train Test Split
>Why this is neccessary do??
```ruby
code in notebook
```

6. Feature Scalling using Z-Score Normalization
>why this is neccessary do?? 
```ruby
code in notebook
```

## Modelling

