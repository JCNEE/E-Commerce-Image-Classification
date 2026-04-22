from __future__ import annotations

import shutil
from pathlib import Path

import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"
MOBILENET_V2_TFLITE_PATH = ARTIFACTS_PATH / "mobilenet_v2_imagenet.tflite"
IMAGENET_CLASS_INDEX_PATH = ARTIFACTS_PATH / "imagenet_class_index.json"
IMAGENET_CLASS_INDEX_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"


def export_mobilenet_v2_tflite() -> None:
	model = MobileNetV2(weights="imagenet")
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	tflite_model = converter.convert()
	MOBILENET_V2_TFLITE_PATH.write_bytes(tflite_model)


def export_imagenet_class_index() -> None:
	cache_path = Path(tf.keras.utils.get_file("imagenet_class_index.json", origin=IMAGENET_CLASS_INDEX_URL))
	shutil.copyfile(cache_path, IMAGENET_CLASS_INDEX_PATH)


def validate_export() -> None:
	interpreter = tf.lite.Interpreter(model_path=str(MOBILENET_V2_TFLITE_PATH))
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()[0]
	output_details = interpreter.get_output_details()[0]
	print(f"Wrote {MOBILENET_V2_TFLITE_PATH.relative_to(PROJECT_ROOT)}")
	print(f"Input tensor: shape={input_details['shape']}, dtype={input_details['dtype']}")
	print(f"Output tensor: shape={output_details['shape']}, dtype={output_details['dtype']}")
	print(f"Wrote {IMAGENET_CLASS_INDEX_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
	ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
	export_mobilenet_v2_tflite()
	export_imagenet_class_index()
	validate_export()


if __name__ == "__main__":
	main()