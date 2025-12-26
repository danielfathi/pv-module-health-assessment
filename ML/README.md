# Classical Machine Learning

This folder contains the classical machine learning part of the project.
The main idea here is to build a simple and interpretable baseline before
moving to deep learning models.

The task is a multi-class image classification problem. Given an image of a
photovoltaic (PV) module, the goal is to identify the surface condition of the
panel and classify it as either healthy or affected by a specific defect.

## Problem setup
The following classes are considered in this project:
- Burn marks
- Corrosion
- Delamination
- Discoloration
- Glass breakage
- Snail trail
- Good panel

The dataset is relatively small but fairly balanced across classes. This is a
common situation in real maintenance applications, where collecting and
labeling large datasets is expensive and time-consuming.

## Dataset
- Total number of images: 211
- Number of classes: 7
- Around 25–35 images per class

The dataset is not included in this repository.

## Preprocessing
All images are resized to a fixed resolution and normalized before feature
extraction. The dataset is split into training and test sets using an
80/20 split.

## Feature extraction
Instead of learning features automatically, handcrafted features are used.
For each image, the following features are extracted:

- Color histograms (RGB)
- Texture features based on GLCM
- Local Binary Patterns (LBP)

All features are combined into a single feature vector.

## Models
Two classical machine learning models are used:

- Support Vector Machine (SVM) with an RBF kernel
- Random Forest classifier

Class weighting and parameter tuning are applied to improve performance and
stability.

## Data augmentation
Because the dataset is small, data augmentation is applied when training the
SVM model. This includes rotations, flips, brightness changes, noise, and
random cutout. The effect of augmentation on classification performance is
explicitly evaluated.

## Evaluation
The models are evaluated using:
- Classification accuracy
- Confusion matrix
- Precision, recall, and F1-score

## Results
The classical machine learning models perform very well on this dataset.
Random Forest achieves up to 100% accuracy, while SVM reaches similar
performance when data augmentation is used.

Even though the results are high, they should be interpreted carefully due to
the limited dataset size.

## Summary
This classical machine learning pipeline provides a strong and interpretable
baseline for PV module defect detection. It shows that with well-designed
features, simple models can be very effective, especially under realistic
industrial constraints. These results are later used as a reference when
comparing with deep learning models.
