"""
prepare_data.py
---------------
Provides preprocessing utilities for MobileNetV2.
"""

import tensorflow as tf

def preprocess_image(img_path):
    """
    Loads and preprocesses an image for MobileNetV2.
    """
    img = tf.keras.utils.load_img(img_path, target_size=(224,224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array
