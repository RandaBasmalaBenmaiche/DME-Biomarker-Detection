# 🧬 Detection, Quantification, and Longitudinal Analysis of Diabetic Macular Edema Biomarkers in OCT Images

A research-oriented deep learning framework for automated analysis of Diabetic Macular Edema (DME) using Optical Coherence Tomography (OCT), focusing on biomarker detection, quantification, and temporal disease monitoring.

---

## 1. Clinical Background

Diabetic Macular Edema (DME) is a vision-threatening complication of diabetic retinopathy and represents one of the leading causes of blindness among diabetic patients worldwide.

Pathophysiologically, DME is characterized by the accumulation of extracellular fluid within the macula due to the breakdown of the blood-retinal barrier. This results in structural alterations of retinal layers, which can be visualized using Optical Coherence Tomography (OCT).

Clinically, the assessment of DME relies on the identification of several key biomarkers:

* **Cystoid Spaces (Intraretinal Fluid)**
  Hyporeflective cavities within the retina indicating fluid accumulation.

* **Disorganization of Retinal Inner Layers (DRIL)**
  Loss of distinguishable boundaries between inner retinal layers, associated with poor visual prognosis.

* **Hyperreflective Dots (HRD)**
  Small, highly reflective foci believed to correspond to lipid exudates, inflammatory cells, or debris.

Accurate identification and monitoring of these biomarkers are essential for:

* Evaluating disease severity
* Guiding treatment decisions
* Assessing therapeutic response over time

However, manual interpretation of OCT images is:

* Time-consuming
* Operator-dependent
* Subject to inter-observer variability

This motivates the development of automated, quantitative, and reproducible analysis systems.

---

## 2. Research Motivation

While most existing works focus on **static detection** of retinal abnormalities, clinical practice requires **longitudinal understanding** of disease evolution.

This project is motivated by the need to move from:

> **“What is present in this scan?”**
> to
> **“How is the disease evolving over time?”**

The proposed system aims to:

* Detect multiple biomarkers automatically
* Quantify their structural properties
* Enable future temporal comparison between patient visits *(ongoing work)*

---

## 3. Objectives

The main objectives of this work are:

1. Develop automated methods for detecting DME-related biomarkers in OCT images
2. Compare multiple deep learning architectures across tasks
3. Quantify biomarker characteristics (area, count, distribution) *(TODO)*
4. Establish a foundation for longitudinal disease monitoring *(TODO)*
5. Design a clinically meaningful severity scoring system *(TODO)*

---

## 4. Methodology

### 4.1 Problem Decomposition

The problem is decomposed into three complementary tasks:

| Biomarker | Task Type      | Modeling Approach                 |
| --------- | -------------- | --------------------------------- |
| Cystoids  | Segmentation   | Pixel-wise prediction             |
| DRIL      | Classification | Image-level binary classification |
| HRD       | Regression     | Heatmap prediction                |

📌 **Methodological Note**
DRIL is modeled as classification rather than segmentation to reduce annotation complexity and improve training feasibility. This choice will be further justified in future work.

---

### 4.2 Model Architectures

#### Segmentation

* U-Net
* U-Net++
* Attention U-Net

#### Classification

* ResNet18
* ResNet50
* EfficientNet-B0

#### Regression (HRD)

* CNN-based heatmap regression *(implementation ongoing)*

---

### 4.3 Training Setup

* Framework: PyTorch
* Input resolution: 512 × 512
* Threshold: 0.5

#### Loss Functions

* Segmentation: Binary Cross-Entropy (BCE)
* Regression: Mean Squared Error (MSE)
* Combined loss: *(TODO: define formal equation)*

---

## 5. Dataset

The datasets used in this project consist of OCT images annotated for **three biomarkers**: Cystoid Spaces, DRIL, and Hyperreflective Dots (HRD). Images were collected from public sources (Kaggle, deep web) and hospital collaborators, and annotations were performed by **two ophthalmologists and one bioinformatics intern**.

All data is **anonymized** and preprocessed to comply with privacy regulations.

---

### 5.1 Cystoid Spaces Segmentation

| Dataset            | Total Images | Annotated Images | Annotation Type |
| :----------------- | :----------- | :--------------- | :-------------- |
| Kaggle OCT dataset | 1460         | 1460             | Unknown         |
| Hospital Dataset 1 | 324          | 128              | Polygon (CVAT)  |
| Hospital Dataset 2 | 75           | 62               | Polygon (CVAT)  |

* **Image size:** 512 × 512
* **Notes:** Only annotated images were used for training/validation. Polygons were converted into binary masks for segmentation.

---

### 6.2 DRIL Classification

| Dataset              | Total Images | Annotated Images | Annotation Type |
| :------------------- | :----------- | :--------------- | :-------------- |
| Deep web OCT dataset | 11,000       | 2,302            | Binary labels   |

* **Class Distribution:**

  * Positive (DRIL present): 795
  * Negative (DRIL absent): 1,507
* **Image size:** 224 × 224
* **Notes:** Weighted loss used to handle class imbalance.

---

### 6.3 Hyperreflective Dots Detection

| Dataset              | Total Images | Annotated Images | Annotation Type   |
| :------------------- | :----------- | :--------------- | :---------------- |
| Deep web OCT dataset | 11,000       | 1,556            | Points → Heatmaps |

* **Image size:** 512 × 512
* **Notes:** Annotations were points manually placed; heatmaps were generated from these points for model training.

---

### 6.4 Preprocessing & Augmentation

All datasets undergo consistent preprocessing to ensure robustness across models:

* Image normalization
* Resizing to 512×512 (Cysts, HRD) or 224×224 (DRIL)
* Data augmentation: rotation, flipping, contrast adjustment
* Noise reduction

---
## 6. Results

### 6.1 Cystoid Segmentation

#### Dataset 72 (Test Set: 10 images)

| Model           | Dice   | IoU    | Precision | Recall |
| --------------- | ------ | ------ | --------- | ------ |
| Attention U-Net | 0.6768 | 0.5115 | 0.8907    | 0.5458 |
| U-Net           | 0.8491 | 0.7378 | 0.8265    | 0.8730 |
| U-Net++         | 0.8495 | 0.7384 | 0.7925    | 0.9153 |

#### Dataset 324 (Test Set: 20 images)

| Model           | Dice       | IoU        | Precision | Recall |
| --------------- | ---------- | ---------- | --------- | ------ |
| Attention U-Net | 0.7145     | 0.5558     | 0.7675    | 0.6683 |
| U-Net           | **0.7645** | **0.6188** | 0.7613    | 0.7677 |
| U-Net++         | 0.7309     | 0.5759     | 0.7371    | 0.7247 |

#### Observations

* U-Net demonstrates the most stable performance across datasets
* U-Net++ increases recall but introduces more false positives
* Attention U-Net shows high precision but low sensitivity

---

### 6.2 DRIL Classification

#### Best Model: ResNet50 (Full Fine-Tuning)

* Accuracy: **0.8818**
* F1-Score: **0.8611**
* ROC-AUC: **0.9315**
* Sensitivity: 0.7167
* Specificity: 0.9692

#### Observations

* High specificity indicates strong performance on healthy cases
* Lower sensitivity highlights difficulty in detecting DRIL
* Model performance is sensitive to training strategy

---

### 6.3 Hyperreflective Dots (HRD)

| Metric | Value |
| ------ | ----- |
| Dice   | TODO  |
| MSE    | TODO  |

#### Observations

* Detection of small structures remains challenging
* Performance highly dependent on annotation quality

---

### 6.4 Key Insights

* Simpler architectures (U-Net) outperform more complex variants
* Small-scale features (HRD) are the most difficult to model
* Class imbalance significantly impacts classification performance
* Trade-offs between precision and recall vary by architecture

---

## 7. Quantification & Clinical Interpretation *(Future Work)*

The project aims to move beyond detection toward **quantitative biomarkers**:

| Biomarker | Planned Metric                    |
| --------- | --------------------------------- |
| Cystoids  | Surface area, count               |
| DRIL      | Binary presence + extent *(TODO)* |
| HRD       | Dot count, intensity              |

📌 Severity scoring function: **TODO**

---

## 8. Longitudinal Analysis *(Future Work)*

A key objective is to enable comparison between OCT scans across time:

* Detect changes in biomarker size and distribution
* Quantify progression or regression
* Provide objective indicators for treatment response

📌 Longitudinal scoring: **TODO**

---

## 9. Limitations

* Limited dataset size
* Class imbalance
* Variability in OCT acquisition conditions
* Absence of clinical validation
* Incomplete quantification framework *(ongoing)*

---

## 10. Future Work

* Define severity scoring formula
* Improve HRD detection (small object problem)
* Integrate multi-biomarker analysis
* Extend to longitudinal patient tracking
* Validate with larger clinical datasets

---

## 11. Implementation

For more details:

👉 See [`/docs`](./docs)

---

## 12. Disclaimer

This project is intended for **research and educational purposes only** and is not approved for clinical use.

---

## 13. Author

**Randa Benmaiche**
AI Student | Computer Vision & Medical Imaging
GitHub: [https://github.com/RandaBasmalaBenmaiche](https://github.com/RandaBasmalaBenmaiche)
