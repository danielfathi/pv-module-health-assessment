# Health Assessment of Photovoltaic Modules Using Image Analysis

This repository contains the implementation of a project developed for the course
**Industrial AI and eMaintenance**.

The objective of this project is to assess the health condition of photovoltaic (PV)
modules using image analysis and to compare classical machine learning and deep learning
approaches under realistic industrial constraints.

## Repository Structure

- **ML/**  
  Classical machine learning pipeline using handcrafted features (Color Histogram, GLCM,
  LBP) and classifiers such as SVM and Random Forest.

- **DeepLearning/**  
  Deep learning models based on transfer learning, including MobileNetV2 and
  EfficientNetB0, with and without fine-tuning.

## Notes
- Datasets and trained model weights are intentionally excluded from this repository
  to keep it lightweight and focused on methodology and reproducibility.
- The project links classification results to practical maintenance decisions
  (keep, repair, replace).

## Author
Daniel Fathi
