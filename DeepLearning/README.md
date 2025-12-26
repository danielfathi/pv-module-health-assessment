# Deep Learning

This folder contains the deep learning part of the project.
Here, the goal is to move from handcrafted features to end-to-end
image learning using convolutional neural networks (CNNs).

This part is a continuation of the previous work, where classical
machine learning models were used as a baseline for comparison.

## Goal
The objective is to automatically detect and classify visible surface
defects in photovoltaic (PV) modules using deep learning and to study
how well these models work when the dataset is small and slightly
imbalanced, which is very common in real industrial maintenance.

The task is a multi-class classification problem with the following
classes:
- Burn Marks
- Corrosion
- Delamination
- Discoloration
- Glass Breakage
- Snail Trail
- Good Panel

## Approach
All deep learning models are trained using transfer learning.
Pre-trained weights from ImageNet are used to benefit from features
learned on large datasets and adapt them to the smaller PV dataset.

Three different CNN architectures are implemented and compared:

- **MobileNetV2**  
  A lightweight and efficient model designed for fast training and
  low computational cost. It is well suited for small datasets and
  limited hardware.

- **EfficientNetB0 (Base)**  
  A compact but powerful architecture that balances accuracy and
  computational cost using compound scaling. Most layers are kept
  frozen.

- **EfficientNetB0 (Fine-Tuned)**  
  The same architecture as above, but with all layers unfrozen and
  trained end-to-end to better adapt to the PV dataset.

## Training and Evaluation
All models are trained on the same dataset and evaluated using:
- Training and validation accuracy
- F1-score
- Confusion matrices

Because the dataset is small and slightly imbalanced, validation
accuracy is considered the main indicator of generalization.

## Results
- **MobileNetV2** reached very high training accuracy and achieved the
  best validation performance among all models (around 46% validation
  accuracy). It showed some overfitting but could still recognize
  certain classes like Burn Marks and Good Panel reasonably well.

- **EfficientNetB0 (Base)** performed poorly. Since most layers were
  frozen, the model could not adapt to the PV dataset and showed clear
  underfitting, with predictions close to random.

- **EfficientNetB0 (Fine-Tuned)** achieved very high training accuracy
  but extremely low validation accuracy. The model heavily overfitted
  and mostly memorized the training data, failing to generalize to new
  images.
