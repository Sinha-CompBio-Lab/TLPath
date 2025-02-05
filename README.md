# TLPath
## Framework for predicting Telomere Length form Whole Slide Image 

**Abstract:** Telomeres are protective nucleoprotein complexes at chromosome ends and their shortening in aged tissues is one of the aging hallmarks. This shortening is associated with various age-related diseases and increased mortality risk. However, we lack high-throughput methods to measure telomere length. Prior studies pointed to underlying connection between telomere length and cellular morphology, but lacks a systematic assessment. We developed TLPath, a computational framework that predicts the length of the telomere from routinely available tissue histopathology (H&E) images. TLPath was trained on a paired dataset with both H&E and telomere length labels via Luminex assay from public GTex cohort, comprising XX patch images, 5,285 whole-slide images, from 926 non-disease individuals spanning 18 tissue. TLPath comprises four-steps: 1) preprocess tissue slides 2) extract morphological features from each patch using UNI, a pretrained foundation model, 3) pool these features to create a whole slide-level representation, and, 4) develop a morphological features-based model to predict tissue telomere length. When TLPath was tested on test GTEx data never seen before to the model, it predicts telomere length with an average correlation = 0.517 across 11 tissue types. In comparison, chronological age predict this with a correlation of 0.12. Beyond predicting telomere length in individuals of a wide age-range, we found that TLPath can also predict telomere length in age-matched samples across all tissues. TLPath’s most important features were nucleus-to-cytoplasmic ratio and variation in nucleus shape. TLPath is the first-in-class digital pathology tool to predict telomere length. enabling large-scale quantification. 	

![alt text](docs/pics/Title_Photo.png)


## Installation
First clone this repo and cd into the directory: 
```
git clone https://github.com/Sinha-CompBio-Lab/TLPath.git
cd TLPath
```

### 1. Get Access 

### 2. Running Inference
