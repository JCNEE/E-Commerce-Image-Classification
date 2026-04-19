"""
train_models.py
---------------
Loads MobileNetV2 pretrained on ImageNet weights.
No training required.
"""

import tensorflow as tf

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
