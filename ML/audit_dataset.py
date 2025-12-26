import os
import matplotlib.pyplot as plt
import random
import cv2

dataset_path = "paneldataset"


class_counts = {}
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        class_counts[class_name] = count


print("Image counts per class:")
for k, v in class_counts.items():
    print(f"{k}: {v}")


plt.figure(figsize=(8,5))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xticks(rotation=45)
plt.ylabel("Number of Images")
plt.title("Class Distribution in Dataset")
plt.show()


for class_name in class_counts.keys():
    class_path = os.path.join(dataset_path, class_name)
    sample_images = random.sample(os.listdir(class_path), min(3, len(os.listdir(class_path))))  # حداکثر ۳ تصویر
    fig, axes = plt.subplots(1, len(sample_images), figsize=(10,3))
    fig.suptitle(class_name, fontsize=14)
    
    for ax, img_name in zip(axes, sample_images):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)[:,:,::-1]  # BGR -> RGB
        ax.imshow(img)
        ax.axis("off")
    plt.show()

