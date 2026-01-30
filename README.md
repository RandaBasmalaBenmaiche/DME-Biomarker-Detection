# 🧬 DME-Biomarker-Detection: Detection, Quantification, and Longitudinal Analysis of Diabetic Macular Edema Biomarkers in OCT Images
A research-driven project exploring deep learning approaches for the detection and quantification of Diabetic Macular Edema biomarkers in OCT images, aiming to support clinical decision-making and improve ophthalmic diagnostics

## 1. Introduction

Diabetic Macular Edema (DME) is one of the most severe complications of diabetes and a leading cause of vision impairment worldwide. In Algeria, the prevalence of this pathology represents a major public health challenge. According to Prof. Nouri, head of department at Beni Messous Hospital, approximately 12% of adult diabetic patients suffer from diabetic retinopathy, and 42% of these patients develop Diabetic Macular Edema, a severe form of diabetic retinopathy. These alarming statistics highlight the magnitude of the healthcare challenge faced by the country.

The management of DME is complex and costly, placing a significant burden on healthcare systems. Moreover, accurately assessing the evolution of the disease over time is essential to determine whether a patient’s treatment is effective or requires adjustment. Traditional clinical workflows rely heavily on manual interpretation of OCT images by ophthalmologists, which is time-consuming and subject to inter-observer variability.

Artificial Intelligence (AI) offers a promising solution to improve early detection, severity assessment, and monitoring of DME. By leveraging deep learning techniques for medical image analysis, AI systems can provide precise, reproducible, and quantitative evaluations of retinal biomarkers. This project proposes an end-to-end deep learning framework for the automated detection, quantification, and temporal comparison of DME-related biomarkers in OCT images, with the objective of supporting ophthalmologists in diagnosis and follow-up of patients.

---

## 2. Motivation and Impact

The ability to monitor disease progression is critical in chronic pathologies such as DME. Beyond detecting biomarkers in a single image, this project introduces a longitudinal analysis approach that compares OCT images acquired at different time points during a patient’s treatment.

By comparing successive OCT scans of the same patient, the system can:

* Determine whether the patient’s condition is improving, worsening, or remaining stable.
* Quantify changes in retinal biomarkers over time.
* Provide objective indicators to support therapeutic decision-making.
* Reduce subjectivity in clinical evaluation.

From both a clinical and research perspective, this project explores the integration of computer vision, medical imaging, and temporal analysis, contributing to the development of intelligent systems capable of dynamic disease monitoring rather than static diagnosis.

---

## 3. Objectives

The main objectives of this project are:

1. To develop an AI-based system for detecting DME-related biomarkers in OCT images.
2. To quantify retinal structural abnormalities associated with DME.
3. To implement a longitudinal comparison module to analyze disease progression over time.
4. To evaluate multiple deep learning architectures for detection, segmentation, and classification tasks.
5. To design a medically meaningful severity scoring function based on expert knowledge.
6. To provide a robust and extensible framework for future research in medical AI.

---

## 4. Proposed AI Solution

### 4.1 Overview of the Approach

The proposed solution analyzes OCT images acquired at different stages of a patient’s treatment. For each patient, the system processes multiple OCT scans and extracts quantitative information about retinal biomarkers. These outputs are then compared across time to determine the evolution of the disease.

The system relies on four major biomarkers to detect and assess the severity of DME:

### 4.2 Biomarkers of Interest

1. **Disorganization of Retinal Inner Layers (DRIL)**
   This biomarker corresponds to the loss of clear boundaries between the inner retinal layers, representing a key indicator of disease progression.

2. **Discontinuity of the Ellipsoid Zone and External Limiting Membrane (ELM)**
   The photoreceptor layer consists of inner segments (ellipsoid zone) and outer segments, separated by the external limiting membrane. Disruptions in these structures are strongly associated with visual impairment.

3. **Cystoid Spaces (Intraretinal Cysts)**
   These appear as dark circular regions in OCT images. Their size, distribution, and volume are critical indicators of edema severity and risk of vision loss.

4. **Hyperreflective Foci**
   These are small bright spots visible in OCT images, often associated with pathological changes in retinal tissues and inflammation.

For each biomarker, specialized deep learning models are selected and optimized for detection and quantification.

---

## 5. Models and Methodology

For each biomarker, different deep learning models are explored:

### 5.1 Object Detection Models

Used mainly for detecting hyperreflective foci and localized abnormalities:

* YOLO (v1–v8)
* SSD (Single Shot MultiBox Detector)
* RetinaNet
* R-CNN family

### 5.2 Segmentation Models

Used for pixel-level segmentation of cystoid spaces and fluid regions:

* U-Net and its variants
* Other encoder–decoder architectures

### 5.3 Classification Models

Used for severity assessment and feature extraction:

* ResNet
* EfficientNet
* Custom CNN architectures

After applying these models, an evaluation function integrates the outputs of different biomarkers to estimate the overall severity of DME. This function is designed based on clinical expertise, enabling a medically meaningful interpretation of AI predictions.

---

## 6. Dataset

### 6.1 Data Description

The dataset consists of OCT retinal images collected from clinical collaborators and publicly available sources. Expert annotations were provided by ophthalmologists to identify key biomarkers associated with DME.

### 6.2 Preprocessing

The following preprocessing steps were applied:

* Image normalization and resizing
* Noise reduction
* Data augmentation (rotation, flipping, contrast enhancement)
* Annotation formatting for deep learning models

Due to ethical and privacy considerations, parts of the dataset may not be publicly accessible.

---

## 7. Longitudinal Analysis: Disease Progression Monitoring

A key contribution of this project is the integration of a longitudinal analysis module.

For each patient, OCT images acquired at different time points are compared to:

* Measure variations in biomarker size, shape, and intensity.
* Quantify progression or regression of retinal abnormalities.
* Generate a temporal profile of disease evolution.

Based on these comparisons, the system provides an interpretable assessment of whether the patient’s condition is improving, deteriorating, or stable, offering valuable insights for personalized treatment monitoring.

---

## 8. Evaluation Metrics

The performance of the models is evaluated using clinically relevant metrics:

* Precision, Recall, and F1-score
* Intersection over Union (IoU)
* Mean Average Precision (mAP)
* Dice Coefficient (for segmentation)
* Accuracy and ROC-AUC (for classification)
* Temporal consistency metrics for longitudinal analysis

---

## 9. Challenges and Limitations

Several challenges were encountered:

* Limited availability of annotated medical data
* Class imbalance among biomarkers
* Variability in OCT image quality
* Generalization across different devices and populations

---

## 10. Future Work

Future directions include:

* Integration of multimodal clinical data (OCT + patient metadata).
* Development of explainable AI methods for clinical interpretation.
* Extension of the framework to other retinal diseases.
* Deployment of the system in real-world clinical environments.
* Validation through large-scale clinical studies.

---

## 11. Disclaimer

This project is intended for research and educational purposes only. It is not designed for clinical use or medical diagnosis yet.

---

## 12. Author

**Randa Benmaiche**
AI Student | Computer Vision & Medical Imaging
GitHub: [https://github.com/your-username](https://github.com/your-username)
ORCID: [https://orcid.org/your-id](https://orcid.org/your-id)

---


