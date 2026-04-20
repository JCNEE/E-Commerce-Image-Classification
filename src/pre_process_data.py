"""
pre_process_data.py
-------------------
Defines which ImageNet classes count as 'Product' (huntable animals).
No dataset download required.
"""
#===================================================================
# START OF PRE-PROCESSING OF THE TEST MODEL
#===================================================================



# 1. Import Libraries
import os
import re
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array

#2.Define Constants
IMAGE_DIR   = "data\Image_Dataset"   
IMG_SIZE    = (224, 224)                     # EfficientNetB0 native input size
BATCH_SIZE  = 32

LABEL_MAP = {
    "1": "buffalo",
    "2": "elephant",
    "3": "rhino",
    "4": "zebra"
}



#3. Load & Parse Images

def load_dataset(image_dir):
    images, labels = [], []

    # Regex to capture the leading digit: "1 (3).jpg" → group(1) = "1"
    pattern = re.compile(r'^(\d+)\s*\(')

    for filename in sorted(os.listdir(image_dir)):
        match = pattern.match(filename)
        if not match:
            continue                          # skip non-matching files

        class_id = match.group(1)
        if class_id not in LABEL_MAP:
            continue                          # skip unknown class IDs

        img_path = os.path.join(image_dir, filename)

        # Load → resize → convert to array → normalise to [0, 1]
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0

        images.append(img_array)
        labels.append(LABEL_MAP[class_id])

    return np.array(images), np.array(labels)

images, labels = load_dataset(IMAGE_DIR)
print(f"Loaded {len(images)} images | Shape: {images.shape}")
print(f"Class distribution: { {l: np.sum(labels==l) for l in np.unique(labels)} }")



#4. Encode Labels

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)         # e.g. buffalo→0, elephant→1, …
labels_onehot  = to_categorical(labels_encoded)   # one-hot for softmax output

print("Classes:", le.classes_)   # confirms alphabetical encoding order



#5. Split of Train/Validation/Test 

# 70% train | 15% validation | 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels_onehot, test_size=0.30, random_state=42, stratify=labels_encoded
)

# Split the 30% temp into equal val and test halves
labels_temp = np.argmax(y_temp, axis=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=labels_temp
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")



#6. Data Augmentation (artificialy expandes dataset to prevent overfitting)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10),
], name="augmentation")




#7.Build tf.data Pipelines (for quick testing)

def make_dataset(X, y, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(X_train, y_train, augment=True)
val_ds   = make_dataset(X_val,   y_val,   augment=False)
test_ds  = make_dataset(X_test,  y_test,  augment=False)



#8. Verify a Sample Batch (Sanity Check)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for images_batch, labels_batch in train_ds.take(1):
    for i, ax in enumerate(axes.flat):
        ax.imshow(images_batch[i].numpy())
        ax.set_title(le.classes_[np.argmax(labels_batch[i])])
        ax.axis("off")
plt.suptitle("Sample Training Images (after augmentation)")
plt.tight_layout()
plt.show()



#9. Save Preprocessed Data to HDF5 (for efficient loading in training script) for EfficientNetB0 model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


with h5py.File(os.path.join(ARTIFACTS_DIR, "wildlife_data.h5"), "w") as f:
    f.create_dataset("X_train", data=X_train, compression="gzip")
    f.create_dataset("y_train", data=y_train)
    f.create_dataset("X_val",   data=X_val,   compression="gzip")
    f.create_dataset("y_val",   data=y_val)
    f.create_dataset("X_test",  data=X_test,  compression="gzip")
    f.create_dataset("y_test",  data=y_test)
    f.create_dataset("classes", data=np.array(le.classes_, dtype="S10"))

print("Saved successfully!")

#===================================================================
# END OF PRE-PROCESSING OF THE TEST MODEL
#===================================================================


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------


#===================================================================
# START OF PRE-PROCESSING OF THE 3 MAIN MODELS
#===================================================================



# List of huntable animals (ImageNet class names)
PRODUCT_CLASSES = [
    "zebra",
    "giraffe",
    "ostrich",
    "antelope",
    "springbok",
    "wildebeest"
]

def get_product_classes():
    """Return the list of product classes."""
    return PRODUCT_CLASSES

if __name__ == "__main__":
    print("Product classes:", get_product_classes())
