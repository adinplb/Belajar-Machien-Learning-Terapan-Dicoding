# First Project Report of Applied ML
>### ***"Breast Tumor Prediction and Diagnosis Using Quantitative Cell Nuclear Phenotype Features in Supervised Machine Learning Algorithms"***
>>#### Issued by **Muhammad Adin Palimbani** 
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
- Sumber Data + Link
- Jumlah Data
- Kondisi Data
- Informasi Mengenai data
- Menguraikan seluruh Fitur
Dataset yang digunakan berasal dari UC Irvine Machine Learning Repository. Pada pryek ini menggunakan dataset berformat csv berikut dimana sesuai dengan topik yang saya ambil yaitu mengenai breast cancer diagnosis. Dataset ini memiliki 564 data yang memiliki 32 fitur numerik dan 1 fitur kategorikal yang akan diubah menjadi integer sebagai target predict. Berikut features nya:
- Radius: The radius of and individual nucleaus is measured by averaging the length of the radial line segments defined by the centeroid of the snake and the individual snake points
- Perimeter
- Area
- Compactness
- Smoothness
- Concavity
- Symmetry
- Fractal Dimension
- Texture

### Exploratory Data Analysis and Visualization 
#### Check Missing Values
#### Outliers
#### Univariate Analysis
#### Multivariate Analysis
####



<https://www.markdownguide.org>
<fake@example.com>
