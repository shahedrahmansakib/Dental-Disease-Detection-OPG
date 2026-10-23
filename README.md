# 🦷 Automated Dental Disease Detection & Pathology Localization from OPG Radiographs

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Deep Learning framework developed for undergraduate thesis research at **Dhaka International University (DIU)**. This repository implements automated multi-class classification and explainable AI (Smooth Grad-CAM) localization for cropped dental Orthopantomogram (OPG) radiographs.

---

## 📌 Abstract & Overview
Accurate automated diagnosis from dental X-rays remains critical due to inter-class similarities and severe class imbalances. This project proposes a robust deep-learning pipeline leveraging state-of-the-art architectures (`EfficientNetV2S`, `DenseNet121`, `ConvNeXtTiny`, and `ResNet50`) trained on a rigorously partitioned dataset (70:15:15 Train-Val-Test split).

---

## 🗂️ Dataset & Class Distribution
The dataset comprises cropped dental OPG images categorized into **5 major clinical classes** (Total Images: 2,529):
1. **Crown** (392 images)
2. **Healthy** (55 images - *Boosted via Class Weighting*)
3. **Missing Teeth** (321 images)
4. **Root Canal Treatment (RCT)** (1195 images)
5. **Root Piece** (566 images)

- **Data Splitting:** Physical stratified splitting with `70% Train`, `15% Validation`, and `15% Test`.
- **Resolution:** Standardized at `384x384` pixels with advanced data augmentation.

---

## 🚀 Key Features & Methodologies
- **Class Imbalance Mitigation:** Custom class weighting with a `5.0x` boost for the rare *Healthy* class.
- **Optimization & Regularization:** Utilized `AdamW` optimizer combined with `CosineDecay` learning rate scheduling and controlled dropout layers.
- **Explainable AI (XAI):** Integrated **Smooth Grad-CAM** with bicubic interpolation and Gaussian blur to visualize pathology localization heatmaps clearly.
- **Comparative Benchmarking:** Multi-model evaluation across DenseNet121, ConvNeXtTiny, and ResNet50.

---

## 📊 Experimental Results

### 1. Model Performance Comparison Table
| Model | Train Acc (%) | Val Acc (%) | Test Acc (%) | Macro ROC-AUC | Mean FPR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EfficientNetV2S (Proposed Base Model)** | 89.82 | 88.92 | 89.01 | 0.9832 | 0.0333 |
| **DenseNet121** | 87.05 | 87.34 | 86.65 | - | 0.0407 |
| **ConvNeXtTiny** | 80.15 | 84.96 | 86.13 | - | 0.0392 |
| **ResNet50** | 91.80 | 87.60 | 85.60 | - | 0.0405 |

### 2. Evaluation Metrics
- **Macro ROC-AUC:** Up to `0.9832` across classes.
- **Healthy Class Recall:** Achieved `1.00` with zero confusion.

---

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shahedrahmansakib/Dental-Disease-Detection-OPG.git
