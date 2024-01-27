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

>_mean, _se (standar error) and _worst (largest): mean of the 3 largest values of these features were computed for each image, resulting in 30 features. For instance: field 3 is mean radius, field 13 is radius_se, field 23 is worst radius.

 . | radius_mean | radius_se | radius_worst | 
 --- | --- | --- | --- | 
Definition | mean of distances from center to points on the perimeter | standard error for the mean of distances from center to points on the perimeter | "worst" or largest mean value for mean of distances from center to points on the perimeter | 
Example | 17.99 | 1.095| 25.38 | 

### Exploratory Data Analysis and Visualization 
#### Check Missing Values
#### Outliers
#### Univariate Analysis
#### Multivariate Analysis
####



<https://www.markdownguide.org>
<fake@example.com>
