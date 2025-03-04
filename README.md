# TLPath
## Framework for predicting Telomere Length form Whole Slide Image 

**Abstract:** Telomeres are protective nucleoprotein complexes at chromosome ends and their shortening in aged tissues is one of the aging hallmarks. This shortening is associated with various age-related diseases and increased mortality risk. However, we lack high-throughput methods to measure telomere length. Prior studies pointed to underlying connection between telomere length and cellular morphology, but lacks a systematic assessment. We developed TLPath, a computational framework that predicts the length of the telomere from routinely available tissue histopathology (H&E) images. TLPath was trained on a paired dataset with both H&E and telomere length labels via Luminex assay from public GTex cohort, comprising XX patch images, 5,285 whole-slide images, from 926 non-disease individuals spanning 18 tissue. TLPath comprises four-steps: 1) preprocess tissue slides 2) extract morphological features from each patch using UNI, a pretrained foundation model, 3) pool these features to create a whole slide-level representation, and, 4) develop a morphological features-based model to predict tissue telomere length. When TLPath was tested on test GTEx data never seen before to the model, it predicts telomere length with an average correlation = 0.517 across 11 tissue types. In comparison, chronological age predict this with a correlation of 0.12. Beyond predicting telomere length in individuals of a wide age-range, we found that TLPath can also predict telomere length in age-matched samples across all tissues. TLPath’s most important features were nucleus-to-cytoplasmic ratio and variation in nucleus shape. TLPath is the first-in-class digital pathology tool to predict telomere length. enabling large-scale quantification. 	

![alt text](docs/pics/Title_Photo.png)


## Installation
First clone this repo and cd into the directory: 
```
git clone https://github.com/Sinha-CompBio-Lab/TLPath.git
cd TLPath
conda env create -f env.yaml
conda activate TLPath
```
### 1. Get Access 
To preprocess and get the UNI features from the H&e slides you need access to UNI model weight. Please follow the instructions [here](https://github.com/mahmoodlab/UNI) to get access to UNI weights. For ease of reproducibility we have provided the  whole slide level features which are mean aggregation of patch level features with this code. You may find it at `{ZENODO_PLACEHOLDER}`

### 2. Running Inference
To run an inference on the UNI features please follow the guide in the notebook `run_inference.ipynb`

### 3. Training TLPath
To train TLPath please follow the notebook `train_TLPath.ipynb`. TLPath can also be trained using CLI. 
Telomere data file can be downloaded from : https://gtexportal.org/home/downloads/egtex/telomeres
`python /tlpath/model.py --telomere-file /path/to/telomere.csv --features_dir /path/to/features --output-dir /path/to/output --config /path/to/config.yaml --tissues-to-skip Tissue1 Tissue2`

- `--telomere-file` → Path to the telomere length data CSV file.
- `--features_dir` → Directory containing patch features.
- `--output-dir (optional)` → Directory to save results and models (default: results/TLPath).
- `--config (optional)` → Path to a YAML configuration file.
- `--tissues-to-skip (optional)` → List of tissues to exclude from analysis.(default: None)
