# Cystoid Spaces (Logettes) Detection & Segmentation

## Overview

Cystoid spaces (logettes) are **dark circular or oval regions** observed in OCT images, representing **fluid accumulation within the retinal layers**. They are a key biomarker of **Diabetic Macular Edema (DME)**.

This project focuses on the **automated segmentation and quantification** of these spaces using a deep learning approach trained on a multi-source dataset, including primary clinical data from Algeria.

-----

## Model & Training

I implemented a **U-Net** architecture with a **ResNet-34** encoder using the `segmentation-models-pytorch` library.

### 1\. Preprocessing & Data Uniformity

To handle the brightness and contrast variations across different OCT machines, I applied a custom normalization strategy:

  * **Target Statistics:** Normalization was performed using a **Target Mean of 0.4** and **Target Std of 0.2**, derived from the Tizi Ouzou hospital datasets.
  * **No Clipping:** During training, `np.clip` was intentionally omitted to preserve diagnostic details in the pixel value distribution tails.
  * **Smart Resizing:** Images were resized to **512x512** using a "Longest Max Size" strategy with padding to maintain the original anatomical aspect ratio.

### 2\. Training Environment

  * **Hardware:** Google Colab (NVIDIA T4 GPU).
  * **Parameters:** 15 Epochs (\~20 mins training time).
  * **Loss Function:** Combination of Dice and Binary Cross-Entropy (BCE).

-----

## Results

The model demonstrated strong generalization capabilities on unseen clinical data from the Tizi Ouzou province.

| Dataset Source | Metric | Value |
| :--- | :--- | :--- |
| **Validation Set** | Mean Dice Score | **0.8478** |
| **Hospital (324)** | Mean Dice / IoU | 0.7424 / 0.6230 |
| **Hospital (75)** | Mean Dice / IoU | 0.6938 / 0.5864 |

### Visual Predictions

Below are 5 random samples comparing the input image, the manual ground truth, and the model's prediction:

![Model Predictions test on TO 72](biomarkers/cystoid_spaces/assets/1.png)
![Model Predictions test on TO 324](biomarkers/cystoid_spaces/assets/2.png)

-----

## Data Sources

### 1\. Hospital Dataset – Tizi Ouzou, Algeria (Primary & Secondary)

  * **Source:** Ophthalmology Department, Tizi Ouzou Province.
  * **Details:** 324 scans (Primary) + 75 scans (Secondary).
  * **Annotations:** Clinically validated for cystoid spaces.

### 2\. Kaggle Dataset

  * **Source:** [Intraretinal Cystoid Fluid OCT Dataset](https://www.kaggle.com/datasets/zeeshanahmed13/intraretinal-cystoid-fluid)
  * **Images:** 1609 OCT scans (Manually annotated).

| Source | Total Scans | Annotated | Full Annotation |
| :--- | :--- | :--- | :--- |
| Tizi Ouzou (Primary) | 324 | 128 | 11 patients |
| Tizi Ouzou (Secondary)| 75 | 72 | N/A |
| Kaggle | 1609 | 1609 | N/A |

-----

## Future Work

  * **Augmentation:** Adding Elastic Transforms and Grid Distortions to increase anatomical robustness.
  * **Architecture:** Experimenting with Attention U-Net and deeper backbones (ResNet-50).
  * **Error Analysis:** Detailed investigation into "false negatives" in images with very small cystoid clusters.

-----

## Access & Ethics

  * **Model:** The best performing model weights (`best_model.pth`) are available in the `/models` directory.
  * **Data Privacy:** Hospital datasets are anonymized and available **strictly upon request** for research purposes.
  * **Contact:** Reach out via email at `randa.cs.acc@gmail.com` for access or collaboration.

-----

## References

1.  Nouri, Pr. et al., “Prevalence and management of diabetic macular edema in Algeria,” *Journal of Ophthalmology*, 2021.
2.  Kaggle: [Intraretinal Cystoid Fluid OCT Dataset](https://www.kaggle.com/datasets/zeeshanahmed13/intraretinal-cystoid-fluid).