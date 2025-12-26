# check_dataset.py
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "paneldataset"  
IMG_SIZE = (224, 224)
BATCH_SIZE = 8

#  data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)


x, y = next(train_gen)
plt.figure(figsize=(10, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(x[i])
    label = list(train_gen.class_indices.keys())[y[i].argmax()]
    plt.title(label)
    plt.axis("off")

plt.tight_layout()
plt.show()
