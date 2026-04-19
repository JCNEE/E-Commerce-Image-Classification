"""
prepare_data.py
---------------
Provides preprocessing utilities for MobileNetV2.
"""

import os
import h5py
import tensorflow as tf


#------------------------------ code to load dataset and preprocess images for the test model------------------------------

#-------This should work, but im not entirely certain-----------------

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

#------------------------------ code to load dataset and preprocess images for the test model------------------------------





def preprocess_image(img_path):
    """
    Loads and preprocesses an image for MobileNetV2.
    """
    img = tf.keras.utils.load_img(img_path, target_size=(224,224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array
