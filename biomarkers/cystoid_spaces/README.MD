# Dataset – Cystoid Spaces (Logettes) Detection

## 1. Overview

Cystoid spaces (logettes) are **dark circular or oval regions** observed in OCT images, representing **fluid accumulation within the retinal layers**. They are a key biomarker of **Diabetic Macular Edema (DME)**, as their presence, size, and distribution correlate with disease severity and the risk of vision loss.

This dataset is curated to support **detection, segmentation, quantification, and longitudinal analysis** of cystoid spaces using state-of-the-art deep learning models. It combines multiple sources to provide diversity and robustness for model training and evaluation.

---

## 2. Data Sources

This dataset is composed of **three complementary sources**:

### 2.1 Hospital Dataset – Tizi Ouzou, Algeria (Primary)

* **Source:** Ophthalmology Department, Tizi Ouzou Province
* **Patients:** 69
* **Images:** 324 OCT scans
* **Anonymization:** Images renamed as `PATIENTNUMBER_OCTSCANNUMBER_POSITION`, where `POSITION` is `b` for bottom-eye images and `t` for top-eye images. Bottom corresponds to one eye and top corresponds to the other eye.
* **Annotations:**

  * 128 OCT scans annotated for cystoid spaces
  * Full annotation for **11 patients** (all their scans)
  * Partial annotation for **24 patients** (some scans left unannotated due to clinical workload constraints)
* **Notes:** This dataset represents **clinically validated annotations** while balancing hospital staff limitations. It is suitable for segmentation, detection, and quantification of cystoid spaces.

### 2.2 Hospital Dataset – Tizi Ouzou, Algeria (Secondary)

* **Source:** Ophthalmology Department, Tizi Ouzou Province
* **Images:** 75 OCT scans
* **Patients:** Unknown
* **Annotations:** 72 scans annotated for cystoid spaces
* **Notes:** This dataset is used as **additional training and validation data**. Annotations were carefully curated to ensure quality despite missing patient metadata.

### 2.3 Kaggle Dataset

* **Source:** [Intraretinal Cystoid Fluid OCT Dataset](https://www.kaggle.com/datasets/zeeshanahmed13/intraretinal-cystoid-fluid)
* **Images:** 1609 OCT scans
* **Annotations:** 333 scans manually annotated for cystoid spaces
* **License:** Attribution-NonCommercial-ShareAlike 4.0 International
* **Notes:** This external dataset provides **large-scale supplementary images** to improve model generalization across varied imaging conditions.

---

## 3. Annotation Summary

| Source                 | Total Scans | Annotated Scans | Full Annotation | Partial Annotation |
| ---------------------- | ----------- | --------------- | --------------- | ------------------ |
| Tizi Ouzou – 324 scans | 324         | 128             | 11 patients     | 24 patients        |
| Tizi Ouzou – 75 scans  | 75          | 72              | N/A             | N/A                |
| Kaggle – 1609 scans    | 1609        | 333             | N/A             | N/A                |

> “Full annotation” indicates that **all scans of the patient were annotated**, while “partial annotation” means **only some scans were annotated**. This limitation reflects the practical constraints of hospital clinical staff.

---

## 4. Data Organization (Recommended)

```
cystoid_spaces/
├── data/
│   ├── hospital_324/
│   │   ├── images/
│   │   └── annotations/
│   ├── hospital_75/
│   │   ├── images/
│   │   └── annotations/
│   └── kaggle/
│       ├── images/
│       └── annotations/
├── notebooks/            # Exploration & preprocessing scripts
├── models/               # Trained or baseline models
├── experiments/          # Training experiments & results
└── README.md             # Dataset description
```

> Keeping annotations and images organized per dataset ensures **clear provenance**, supports reproducibility, and respects licensing requirements.

---

## 5. Preprocessing Guidelines

* **Normalization:** Standardize image intensities across datasets
* **Resizing:** Resize images for deep learning input dimensions
* **Denoising:** Apply median or Gaussian filtering to reduce OCT artifacts
* **Data Augmentation:** Flipping, rotation, elastic deformation, brightness/contrast adjustment
* **Annotation Conversion:** CVAT XML → COCO JSON or masks for segmentation

---

## 6. Usage Recommendations

* **Segmentation Models:** U-Net, Attention U-Net, DeepLabV3+, nnU-Net
* **Detection Models:** YOLOv8-seg, Mask R-CNN
* **Quantification:** Measure cystoid area, volume, and intensity
* **Longitudinal Analysis:** Track cystoid metrics over time to monitor disease progression or treatment response

---

## 7. Evaluation Metrics

* **Segmentation:** Dice coefficient, IoU, pixel-wise precision & recall
* **Detection:** mAP (mean Average Precision), precision, recall, F1-score
* **Longitudinal Quantification:** Changes in cystoid area or volume over time

---

## 8. Ethical Considerations and Licensing

* **Hospital Datasets:** Anonymized and used with patient consent for research purposes
* **Kaggle Dataset:** Non-commercial research use under [Attribution-NonCommercial-ShareAlike 4.0 International](https://www.kaggle.com/datasets/zeeshanahmed13/intraretinal-cystoid-fluid)
* **Disclaimer:** Intended **strictly for research and educational purposes**, not for clinical diagnosis

---

## 9. References

1. Nouri, Pr. et al., “Prevalence and management of diabetic macular edema in Algeria,” *Journal of Ophthalmology*, 2021
2. Kaggle: [Intraretinal Cystoid Fluid OCT Dataset](https://www.kaggle.com/datasets/zeeshanahmed13/intraretinal-cystoid-fluid)

