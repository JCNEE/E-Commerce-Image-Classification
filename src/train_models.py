"""
train_models.py
---------------
Loads MobileNetV2 pretrained on ImageNet weights.
No training required.
"""

import os
import h5py

import tensorflow as tf

#--------------------------------------
#should hopefully load the correct data from the h5 file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")

with h5py.File(os.path.join(ARTIFACTS_DIR, "wildlife_data.h5"), "r") as f:
    X_train = f["X_train"][:]
    y_train = f["y_train"][:]
    X_val   = f["X_val"][:]
    y_val   = f["y_val"][:]
    X_test  = f["X_test"][:]
    y_test  = f["y_test"][:]
    classes = [c.decode("utf-8") for c in f["classes"][:]]

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"Classes: {classes}")

#-------------------------------------




def load_mobilenet():
    """
    Loads MobileNetV2 pretrained on ImageNet.
    """
    model = tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3),
        include_top=True,
        weights="imagenet"
    )
    return model

if __name__ == "__main__":
    model = load_mobilenet()
    print("MobileNetV2 loaded with ImageNet weights.")
