"""
Testing purpose
-----------------
Runs a test classification using MobileNetV2 pretrained on ImageNet.
Maps predictions into Product vs Non-Product.
"""

import tensorflow as tf
from prepare_data import preprocess_image
from train_models import load_mobilenet
from pre_process_data import get_product_classes

# Load model and product classes
model = load_mobilenet()
PRODUCT_CLASSES = get_product_classes()

def classify_image(img_path):
    """
    Classify an image as Product (huntable animal) or Non-Product.
    """
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]

    for (_, class_name, prob) in decoded:
        if class_name in PRODUCT_CLASSES:
            return f"Product (huntable animal) — {class_name} ({prob:.2f})"
    return f"Non-Product — {decoded[0][1]} ({decoded[0][2]:.2f})"

if __name__ == "__main__":
    test_image = "app/assets/blue-wildebeest.jpg"         #Change this name to the filepath of the image you want to test
    result = classify_image(test_image)
    print(result)
