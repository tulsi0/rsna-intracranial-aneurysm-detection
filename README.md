# RSNA Intracranial Aneurysm Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![Medical Imaging](https://img.shields.io/badge/Domain-Medical%20Imaging-red)
![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)

---

# Project Overview

This project focuses on **detecting intracranial aneurysms from brain imaging scans using deep learning**.

The dataset consists of **medical imaging data stored in DICOM format** along with metadata containing patient information and aneurysm location labels.

The goal is to build an **end-to-end machine learning pipeline** that:

1. Processes raw medical imaging data  
2. Reconstructs 3D brain volumes  
3. Applies image preprocessing techniques  
4. Trains deep learning models for aneurysm detection  

This system could assist **radiologists and healthcare professionals** in analyzing brain scans.

---

# Dataset

Dataset: **RSNA Intracranial Aneurysm Detection**

The dataset contains:

## Metadata

`train.csv` contains information such as:

- PatientAge  
- PatientSex  
- Modality  
- SeriesInstanceUID  
- Artery-specific aneurysm labels  
- Overall aneurysm presence  

---

## Medical Images

Brain scans are stored as **DICOM image series**.

Example dataset structure:

```
dataset/
│
├── train.csv
│
└── series/
    ├── SeriesInstanceUID_1/
    │      slice1.dcm
    │      slice2.dcm
    │      ...
    │
    ├── SeriesInstanceUID_2/
```

Each folder represents **one brain scan consisting of multiple slices**.

---

# Project Pipeline

The workflow for this project is structured as follows:

```
Raw DICOM Images
        ↓
Exploratory Data Analysis
        ↓
DICOM Image Inspection
        ↓
3D Volume Reconstruction
        ↓
Image Preprocessing
        ↓
Dataset Preparation
        ↓
Deep Learning Model Training
        ↓
Model Evaluation
        ↓
Inference Pipeline
```

---

# Exploratory Data Analysis

EDA was performed to understand the dataset and identify patterns.

Key analyses performed:

- Age distribution of patients  
- Gender distribution  
- Imaging modality distribution  
- Aneurysm presence across arteries  
- Age vs aneurysm occurrence  
- Gender vs aneurysm presence  

Visualization techniques used:

- Histograms
- Count plots
- Box plots
- Correlation heatmaps

These insights help guide **feature understanding and modeling decisions**.

---

# DICOM Image Processing

Medical images are stored in **DICOM format**, which contains:

- Image data
- Medical metadata
- Scan parameters

Processing steps:

1. Load DICOM slices using `pydicom`
2. Extract pixel arrays
3. Sort slices using `InstanceNumber`
4. Stack slices into a **3D volume**
5. Normalize pixel intensities
6. Resize images for model input

Example volume:

```
Original Volume Shape: (150, 512, 512)
Processed Volume Shape: (128, 128, 128)
```

---

# 3D Volume Reconstruction

Each brain scan consists of multiple 2D slices.

These slices are stacked to create a **3D representation of the brain**.

Volume format:

```
Volume = (Depth, Height, Width)
```

Example:

```
(128, 128, 128)
```

This format is required for **deep learning models that analyze volumetric medical data**.

---

# Image Preprocessing

To prepare the dataset for deep learning, several preprocessing steps were applied.

## Slice Sorting
Ensures slices are arranged in correct anatomical order.

## Intensity Normalization

```python
normalized = (img - min) / (max - min)
```

## Standardization

```python
standardized = (img - mean) / std
```

## Image Resizing

All slices are resized to:

```
128 × 128
```

## Volume Construction

Slices are stacked into a **3D tensor**:

```
(D, H, W)
```

---

# Custom DICOM Preprocessor

A custom preprocessing class was implemented to automate the workflow.

Capabilities:

- Load DICOM series
- Handle both 2D and 3D scans
- Sort slices using spatial metadata
- Apply normalization and scaling
- Resize slices
- Construct standardized 3D volumes

Target input shape for models:

```
(32, 384, 384)
```

---

# Deep Learning (Work in Progress)

The modeling pipeline is currently under development.

Planned approach:

- Framework: **PyTorch**
- Model architecture: **EfficientNet**
- Input channels: **3D volume slices**
- Training method: **K-Fold Cross Validation**

Target outputs:

- 13 artery-specific aneurysm predictions
- Overall aneurysm presence

---

# Technologies Used

### Programming Language

```
Python
```

### Libraries

```
NumPy
Pandas
Matplotlib
Seaborn
Pydicom
OpenCV
SciPy
PyTorch
Albumentations
Timm
```

### Environment

```
Kaggle Notebook
GPU Acceleration
```

---

# Project Structure

```
project/
│
├── train.csv
├── series/
│
├── notebooks/
│     eda.ipynb
│     preprocessing.ipynb
│
├── preprocessing/
│     dicom_preprocessor.py
│
├── models/
│
├── training/
│     train_model.py
│
└── README.md
```

---

# Current Progress

Completed:

- Dataset exploration
- Exploratory Data Analysis
- DICOM visualization
- MRI slice inspection
- 3D volume reconstruction
- Image preprocessing pipeline

Currently Working On:

- Deep learning model training
- Cross-validation pipeline
- Model evaluation
- Inference pipeline

---

# Future Improvements

Potential improvements include:

- 3D CNN architectures
- Transformer-based models
- Advanced data augmentation
- Multi-scale inputs
- Ensemble learning

---

# Goal

The goal of this project is to develop a **robust AI-based system capable of detecting intracranial aneurysms from brain scans**, helping support early diagnosis and medical decision-making.
