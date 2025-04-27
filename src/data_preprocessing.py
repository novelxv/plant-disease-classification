import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATASET_DIR = "data"
OUTPUT_DIR = f"{DATASET_DIR}/preprocessed"

IMG_SIZE = (128, 128)
CLASS_NAMES = sorted(os.listdir(DATASET_DIR))
NUM_CLASSES = len(CLASS_NAMES)

def load_data(dataset_dir: str = DATASET_DIR, img_size: tuple = IMG_SIZE) -> tuple:
    """ Load images and labels from dataset directory """
    images, labels = [], []
    for label, class_name in enumerate(CLASS_NAMES):
        if class_name != "preprocessed":
            print("Loading class:", class_name)
            class_dir = os.path.join(dataset_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = Image.open(img_path).convert("RGB").resize(img_size)
                    images.append(np.array(img))
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image: {img_path}")
                    print(e)
    return np.array(images), np.array(labels)

# Load dataset
print("Loading dataset...")
images, labels = load_data()
images = images / 255.0
images = images.astype("float32")

# Split dataset into training and testing sets
print("Splitting dataset...")
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set: {X_train.shape[0]} images")
print(f"Validation set: {X_val.shape[0]} images")
print(f"Testing set: {X_test.shape[0]} images")

# Save preprocessed data
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

print("Preprocessing complete and data saved!")