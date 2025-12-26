# train_efficientnetb0_finetuned.py
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

DATA_DIR = "paneldataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25  #  fine-tuning 

# -------------------------------
#  : Data Augmentation 
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# -------------------------------
#  EfficientNetB0
# -------------------------------
base_model = EfficientNetB0(weights="imagenet", include_top=False,
                             input_shape=IMG_SIZE + (3,))

#  fine-tuning 
for layer in base_model.layers:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(train_gen.num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

# -------------------------------

# -------------------------------
model.compile(optimizer=Adam(learning_rate=3e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# -------------------------------
#  callbacks
# -------------------------------
checkpoint = ModelCheckpoint("efficientnetb0_finetuned_best.h5", monitor="val_accuracy",
                             save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True)

# -------------------------------
# -------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop]
)

# -------------------------------
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("EfficientNetB0 Fine-tuned Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("efficientnetb0_finetuned_accuracy_curve.png", bbox_inches="tight")
plt.show()

# -------------------------------
# -------------------------------
val_gen.reset()
pred = model.predict(val_gen)
y_true = val_gen.classes
y_pred = pred.argmax(axis=1)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys())))

# -------------------------------
#  Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(val_gen.class_indices.keys()),
            yticklabels=list(val_gen.class_indices.keys()))
plt.title("Confusion Matrix - EfficientNetB0 Fine-tuned")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("efficientnetb0_finetuned_confusion.png", bbox_inches="tight")
plt.show()
