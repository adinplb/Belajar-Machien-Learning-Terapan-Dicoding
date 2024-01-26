# First Project Report of Applied ML
### "Breast Tumor Prediction and Diagnosis Using Quantitative Cell Nuclear Phenotype Features in Machine Learning Algorithms"
#### Issued by Muhammad Adin Palimbani
## Project Domain 
1. Issue Focus? <br>
The significant advances in cancer research over the past decades has been carried out with the advent of new technologies in the field of medicine. Scientists have conducted a new approach with different methods for the early prediction of cancer treatment outcome particularly Breast Cancer. One of the example approaches applied is the growing trend on Machine Learning Techniques. However, a common problem in several research is the lack of external validation or testing regarding the predictive performance of their models. This may lead to malformed prediction models and system failures at the production stage

2. Why does the issue need to be resolved? <br>
The accurate prediction models of a disease outcome is extremely depends on the medical data of the patient. Medical data contains the patient's details condition and diagnosis which hold unnecessary and interrelated data. Those data is high dimensional data as well in particular the integration of clinical and genomic mixed data. In several studies, scientists have proved that approaches related to the genomic characteristics provides promising results for cancer detection and identification, for instances, digitized image of a fine needle aspirate (FNA) of a breast mass which represent cell nuclear characteristics in Breast Tumor. However, these methods suffer from low sensitivity regarding their use in screening at early stages and difficulty to determine benign from malignant tumors. This is the reason why the cancer predictive performance models issue need to be resolved in order to prevent malformed prediction and system failures. 

3. How to address the issue? <br>
Interactive image processing techniques, along with a linear programming based inductive classifier, have been used to creeate a highly accurate systm for diagnosis of Breast Tumors. A small fraction of a Fine Needle Aspirate Slide (FNA) is selected and digitized. The digitized image of a FNA of a Breast mass describe chracteristics of the cell nuclei present in the image. Those are computed and become features for this research. So we could possibily diagnosed whether it is Malignet or Benign through Nuclear Feature Extraction.


3. Related research results or refrences from credible sources and clear author. <br> 
W. Nick Street, William H. Wolberg, et al. Nuclear Feature Extraction For Breast Tumor Diagnosis. Computer Science Technical Report #1131, December 1992

## Data Understanding
- Radius: The radius of and individual nucleaus is measured by averaging the length of the radial line segments defined by the centeroid of the snake and the individual snake points
- Perimeter
- Area
- Compactness
- Smoothness
- Concavity
- Symmetry
- Fractal Dimension
- Texture
