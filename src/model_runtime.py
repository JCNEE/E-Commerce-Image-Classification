from __future__ import annotations

import importlib
import json
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"
IMAGENET_CLASS_INDEX_PATH = ARTIFACTS_PATH / "imagenet_class_index.json"
MOBILENET_V2_TFLITE_PATH = ARTIFACTS_PATH / "mobilenet_v2_imagenet.tflite"
IMAGENET_CLASS_INDEX_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"

if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app.animal_catalog import (  # noqa: E402
	get_ranch_animal,
	infer_catalog_animal,
	infer_catalog_animal_id,
	normalise_animal_text,
)


CATALOG_ID_TO_MODEL_LABEL = {
	"buffalo": "buffalo",
	"cape-giraffe": "giraffe",
	"common-ostrich": "ostrich",
	"elephant": "elephant",
	"greater-kudu": "kudu",
	"hippopotamus": "hippopotamus",
	"lion": "lion",
	"plains-zebra": "zebra",
	"rhinoceros": "rhino",
	"springbok": "springbok",
}
MODEL_LABEL_TO_CATALOG_ID = {label: animal_id for animal_id, label in CATALOG_ID_TO_MODEL_LABEL.items()}

MODEL_LABEL_ALIASES = {
	"african elephant": "elephant",
	"african lion": "lion",
	"buffalo": "buffalo",
	"elephant": "elephant",
	"gazelle": "springbok",
	"giraffe": "giraffe",
	"greater kudu": "kudu",
	"hippo": "hippopotamus",
	"hippopotamus": "hippopotamus",
	"indian elephant": "elephant",
	"kudu": "kudu",
	"lion": "lion",
	"ostrich": "ostrich",
	"plains zebra": "zebra",
	"rhino": "rhino",
	"rhinoceros": "rhino",
	"springbok": "springbok",
	"water buffalo": "buffalo",
	"zebra": "zebra",
}

KNOWN_OUTSIDE_SPECIES = {
	"cheetah": "Cheetah",
	"crocodile": "Crocodile",
	"hyena": "Hyena",
	"leopard": "Leopard",
	"penguin": "Penguin",
	"shark": "Shark",
	"warthog": "Warthog",
	"whale": "Whale",
}


@dataclass(frozen=True)
class RuntimePrediction:
	animal_id: str | None
	confidence: float
	mode_label: str
	note: str
	reasons: tuple[str, ...]
	top_candidates: tuple[tuple[str, float], ...] = ()
	catalog_breakdown: tuple[tuple[str, float], ...] = ()
	raw_label: str | None = None
	detected_species: str | None = None


@dataclass(frozen=True)
class LiteRTResources:
	display_name: str
	target_size: tuple[int, int]
	interpreter: object
	input_details: dict[str, object]
	output_details: dict[str, object]
	class_index: dict[int, tuple[str, str]]


class ModelRuntimeUnavailable(RuntimeError):
	pass


def format_species_label(value: str | None) -> str | None:
	if not value:
		return None
	normalised = normalise_animal_text(value)
	if not normalised:
		return None
	return " ".join(part.capitalize() for part in normalised.split())


def resolve_sold_model_label(value: str | None) -> str | None:
	normalised = normalise_animal_text(value)
	if not normalised:
		return None

	if normalised in MODEL_LABEL_ALIASES:
		return MODEL_LABEL_ALIASES[normalised]

	animal_id = infer_catalog_animal_id(normalised)
	if animal_id:
		return CATALOG_ID_TO_MODEL_LABEL.get(animal_id)

	return None


def infer_outside_species_name(value: str | None) -> str | None:
	normalised = normalise_animal_text(value)
	if not normalised:
		return None

	if normalised in KNOWN_OUTSIDE_SPECIES:
		return KNOWN_OUTSIDE_SPECIES[normalised]

	haystack = f" {normalised} "
	for keyword, display_name in KNOWN_OUTSIDE_SPECIES.items():
		if f" {keyword} " in haystack:
			return display_name

	return None


def _load_litert_interpreter_class():
	last_error: Exception | None = None
	for module_name in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
		try:
			module = importlib.import_module(module_name)
			return getattr(module, "Interpreter")
		except Exception as exc:
			last_error = exc

	raise ModelRuntimeUnavailable(
		"LiteRT is not installed in the current environment, so the TFLite website runtime cannot be loaded."
	) from last_error


def _load_litert_interpreter(model_path: Path) -> tuple[object, dict[str, object], dict[str, object]]:
	if not model_path.exists():
		raise ModelRuntimeUnavailable(
			f"The LiteRT model file {model_path.name} was not found in artifacts/. Run src/model_runtime.py to export it first."
		)

	interpreter_class = _load_litert_interpreter_class()

	try:
		interpreter = interpreter_class(model_path=str(model_path))
		interpreter.allocate_tensors()
		input_details = interpreter.get_input_details()[0]
		output_details = interpreter.get_output_details()[0]
	except Exception as exc:
		raise ModelRuntimeUnavailable(
			f"The LiteRT model file at {model_path.name} could not be loaded by the interpreter."
		) from exc

	return interpreter, input_details, output_details


def _normalise_quantization(details: dict[str, object]) -> tuple[float, int]:
	quantization = details.get("quantization")
	if isinstance(quantization, tuple) and len(quantization) == 2:
		scale = float(quantization[0] or 0.0)
		zero_point = int(quantization[1] or 0)
		return scale, zero_point
	return 0.0, 0


def _prepare_image_array(image_bytes: bytes, target_size: tuple[int, int]):
	import numpy as np  # noqa: PLC0415
	from PIL import Image  # noqa: PLC0415

	resampling = getattr(Image, "Resampling", Image).BILINEAR
	with Image.open(BytesIO(image_bytes)) as image:
		resized_image = image.convert("RGB").resize((target_size[1], target_size[0]), resample=resampling)
		image_array = np.asarray(resized_image, dtype=np.float32)

	return np.expand_dims(image_array, axis=0)


def _prepare_mobilenet_v2_tflite_batch(image_bytes: bytes, target_size: tuple[int, int]):
	image_batch = _prepare_image_array(image_bytes, target_size)
	return (image_batch / 127.5) - 1.0


def _set_litert_input_tensor(interpreter: object, input_details: dict[str, object], image_batch) -> None:
	import numpy as np  # noqa: PLC0415

	tensor_dtype = input_details["dtype"]
	prepared_batch = np.asarray(image_batch, dtype=np.float32)

	if tensor_dtype is np.float32:
		interpreter.set_tensor(input_details["index"], prepared_batch.astype(np.float32))
		return

	scale, zero_point = _normalise_quantization(input_details)
	if not scale:
		raise ModelRuntimeUnavailable(
			"The LiteRT backend input tensor is quantized, but the model does not expose valid quantization metadata."
		)

	quantized_batch = np.round(prepared_batch / scale + zero_point)
	if np.issubdtype(tensor_dtype, np.integer):
		limits = np.iinfo(tensor_dtype)
		quantized_batch = np.clip(quantized_batch, limits.min, limits.max)

	interpreter.set_tensor(input_details["index"], quantized_batch.astype(tensor_dtype))


def _get_litert_output_tensor(interpreter: object, output_details: dict[str, object]):
	import numpy as np  # noqa: PLC0415

	output_tensor = interpreter.get_tensor(output_details["index"])
	tensor_dtype = output_details["dtype"]
	if tensor_dtype is np.float32:
		return output_tensor.astype(np.float32)

	scale, zero_point = _normalise_quantization(output_details)
	if not scale:
		return output_tensor.astype(np.float32)

	return scale * (output_tensor.astype(np.float32) - zero_point)


@lru_cache(maxsize=1)
def _load_imagenet_class_index() -> dict[int, tuple[str, str]]:
	if not IMAGENET_CLASS_INDEX_PATH.exists():
		raise ModelRuntimeUnavailable(
			"The ImageNet class index artifact is missing. Run src/model_runtime.py to export it first."
		)

	try:
		with IMAGENET_CLASS_INDEX_PATH.open("r", encoding="utf-8") as class_index_file:
			payload = json.load(class_index_file)
	except (OSError, json.JSONDecodeError) as exc:
		raise ModelRuntimeUnavailable("The ImageNet class index artifact could not be read by the LiteRT backend.") from exc

	class_index: dict[int, tuple[str, str]] = {}
	for index, raw_entry in payload.items():
		if not isinstance(raw_entry, list | tuple) or len(raw_entry) != 2:
			continue
		class_index[int(index)] = (str(raw_entry[0]), str(raw_entry[1]))

	if not class_index:
		raise ModelRuntimeUnavailable("The ImageNet class index artifact is empty or invalid.")

	return class_index


@lru_cache(maxsize=1)
def load_mobilenet_v2_tflite_resources() -> LiteRTResources:
	interpreter, input_details, output_details = _load_litert_interpreter(MOBILENET_V2_TFLITE_PATH)
	return LiteRTResources(
		display_name="MobileNetV2 LiteRT runtime",
		target_size=(224, 224),
		interpreter=interpreter,
		input_details=input_details,
		output_details=output_details,
		class_index=_load_imagenet_class_index(),
	)


def _decode_mobilenet_v2_tflite_predictions(image_bytes: bytes) -> tuple[str, list[tuple[str, str, float]]]:
	import numpy as np  # noqa: PLC0415

	resources = load_mobilenet_v2_tflite_resources()
	image_batch = _prepare_mobilenet_v2_tflite_batch(image_bytes, resources.target_size)
	_set_litert_input_tensor(resources.interpreter, resources.input_details, image_batch)

	try:
		resources.interpreter.invoke()
	except Exception as exc:
		raise ModelRuntimeUnavailable("The image could not be processed by the MobileNetV2 LiteRT runtime.") from exc

	output_tensor = _get_litert_output_tensor(resources.interpreter, resources.output_details)
	if output_tensor.ndim != 2 or output_tensor.shape[0] != 1:
		raise ModelRuntimeUnavailable("The LiteRT backend returned an unexpected output tensor shape.")

	class_scores = output_tensor[0]
	top_indices = np.argsort(class_scores)[-5:][::-1]
	decoded_predictions: list[tuple[str, str, float]] = []
	for index in top_indices:
		class_id, label = resources.class_index.get(int(index), (str(index), f"class-{index}"))
		decoded_predictions.append((class_id, label, float(class_scores[index])))

	return resources.display_name, decoded_predictions


def _build_top_candidate_summary(decoded_predictions: list[tuple[str, str, float]]) -> tuple[tuple[str, float], ...]:
	top_candidates: list[tuple[str, float]] = []
	for _, label, score in decoded_predictions:
		display_label = format_species_label(label) or str(label)
		top_candidates.append((display_label, max(0.0, float(score))))
	return tuple(top_candidates)


def _build_catalog_breakdown(decoded_predictions: list[tuple[str, str, float]]) -> tuple[tuple[str, float], ...]:
	sold_score = 0.0
	outside_score = 0.0
	for _, label, score in decoded_predictions:
		if resolve_sold_model_label(label):
			sold_score += max(0.0, float(score))
		else:
			outside_score += max(0.0, float(score))

	breakdown: list[tuple[str, float]] = []
	if sold_score > 0:
		breakdown.append(("Sold catalog cues", sold_score))
	if outside_score > 0:
		breakdown.append(("Outside catalog cues", outside_score))
	if not breakdown:
		breakdown.append(("Unresolved cues", 1.0))
	return tuple(breakdown)


def _predict_with_tflite_runtime(image_bytes: bytes, file_name: str | None = None) -> RuntimePrediction:
	try:
		display_name, decoded_predictions = _decode_mobilenet_v2_tflite_predictions(image_bytes)
	except ModelRuntimeUnavailable:
		raise
	except Exception as exc:
		raise ModelRuntimeUnavailable("The image could not be processed by the TFLite website runtime.") from exc

	if not decoded_predictions:
		raise ModelRuntimeUnavailable("The TFLite website runtime returned no predictions.")

	top_candidates = _build_top_candidate_summary(decoded_predictions)
	catalog_breakdown = _build_catalog_breakdown(decoded_predictions)
	top_class_id, top_label, top_score = decoded_predictions[0]
	top_display_label = format_species_label(top_label) or top_label
	reasons = [
		f"Prediction backend: {display_name}.",
		f"Top label: {top_display_label} ({top_score:.0%}).",
	]

	resolved_sold_label = resolve_sold_model_label(top_label)
	if resolved_sold_label:
		animal_id = MODEL_LABEL_TO_CATALOG_ID[resolved_sold_label]
		animal = get_ranch_animal(animal_id)
		return RuntimePrediction(
			animal_id=animal_id,
			confidence=max(0.52, top_score),
			mode_label=display_name,
			note="",
			reasons=tuple(reasons),
			top_candidates=top_candidates,
			catalog_breakdown=catalog_breakdown,
			raw_label=top_class_id,
			detected_species=animal.name if animal is not None else top_display_label,
		)

	file_name_match = infer_catalog_animal(file_name)
	if file_name_match is not None:
		reasons.append(
			"The LiteRT prediction did not land on a sold class, but the upload name matched one of the configured sold-animal aliases."
		)
		return RuntimePrediction(
			animal_id=file_name_match.animal_id,
			confidence=0.68,
			mode_label="Filename-assisted fallback",
			note=(
				"The LiteRT output did not resolve cleanly to one of the sold classes, so the site fell back to the upload name."
			),
			reasons=tuple(reasons),
			top_candidates=top_candidates,
			catalog_breakdown=catalog_breakdown,
			raw_label=top_class_id,
			detected_species=file_name_match.name,
		)

	detected_species = infer_outside_species_name(file_name) or infer_outside_species_name(top_label) or top_display_label
	reasons.append("The top LiteRT label did not map to any configured sold-animal class, so the upload is treated as not sold.")
	return RuntimePrediction(
		animal_id=None,
		confidence=max(0.51, top_score),
		mode_label=display_name,
		note=(
			"Anything outside the configured kudu, springbok, giraffe, buffalo, rhino, zebra, ostrich, elephant, lion, and hippopotamus list is routed to not sold."
		),
		reasons=tuple(reasons),
		top_candidates=top_candidates,
		catalog_breakdown=catalog_breakdown,
		raw_label=top_class_id,
		detected_species=detected_species,
	)


def predict_with_runtime(image_bytes: bytes, file_name: str | None = None) -> RuntimePrediction:
	return _predict_with_tflite_runtime(image_bytes=image_bytes, file_name=file_name)


def _load_tensorflow_export_dependencies():
	try:
		import tensorflow as tf  # noqa: PLC0415
		from keras.applications.mobilenet_v2 import MobileNetV2  # noqa: PLC0415
	except Exception as exc:
		raise ModelRuntimeUnavailable(
			"TensorFlow and Keras are required to export the MobileNetV2 TFLite asset. Install them in an export environment before running src/model_runtime.py as a script."
		) from exc

	return tf, MobileNetV2


def export_mobilenet_v2_tflite() -> Path:
	tf, MobileNetV2 = _load_tensorflow_export_dependencies()
	ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
	model = MobileNetV2(weights="imagenet")
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	tflite_model = converter.convert()
	MOBILENET_V2_TFLITE_PATH.write_bytes(tflite_model)
	return MOBILENET_V2_TFLITE_PATH


def export_imagenet_class_index() -> Path:
	tf, _ = _load_tensorflow_export_dependencies()
	ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
	cache_path = Path(tf.keras.utils.get_file("imagenet_class_index.json", origin=IMAGENET_CLASS_INDEX_URL))
	shutil.copyfile(cache_path, IMAGENET_CLASS_INDEX_PATH)
	return IMAGENET_CLASS_INDEX_PATH


def validate_exported_assets() -> None:
	tf, _ = _load_tensorflow_export_dependencies()
	interpreter = tf.lite.Interpreter(model_path=str(MOBILENET_V2_TFLITE_PATH))
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()[0]
	output_details = interpreter.get_output_details()[0]
	class_index = _load_imagenet_class_index()
	print(f"Wrote {MOBILENET_V2_TFLITE_PATH.relative_to(PROJECT_ROOT)}")
	print(f"Input tensor: shape={input_details['shape']}, dtype={input_details['dtype']}")
	print(f"Output tensor: shape={output_details['shape']}, dtype={output_details['dtype']}")
	print(f"Wrote {IMAGENET_CLASS_INDEX_PATH.relative_to(PROJECT_ROOT)}")
	print(f"Loaded {len(class_index)} ImageNet labels")


def export_runtime_assets() -> None:
	export_mobilenet_v2_tflite()
	export_imagenet_class_index()
	_load_imagenet_class_index.cache_clear()
	load_mobilenet_v2_tflite_resources.cache_clear()
	validate_exported_assets()


def main() -> None:
	export_runtime_assets()


if __name__ == "__main__":
	main()
