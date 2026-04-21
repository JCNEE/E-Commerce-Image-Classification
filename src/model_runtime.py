from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"
BEST_MODEL_INFO_PATH = ARTIFACTS_PATH / "best_fine_tuned_model.json"
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app.animal_catalog import (  # noqa: E402
	RANCH_ANIMALS,
	get_ranch_animal,
	infer_catalog_animal,
	infer_catalog_animal_id,
	normalise_animal_text,
)


MODEL_BACKEND_ALIASES = {
	"mobilenet": "mobilenet_v2",
	"mobilenetv2": "mobilenet_v2",
	"mobilenet_v2": "mobilenet_v2",
	"mobilenet-v2": "mobilenet_v2",
	"mobilenetv3": "mobilenet_v3_small",
	"mobilenet_v3": "mobilenet_v3_small",
	"mobilenet-v3": "mobilenet_v3_small",
	"mobilenet_v3_small": "mobilenet_v3_small",
	"mobilenet-v3-small": "mobilenet_v3_small",
	"project_model": "project_model",
	"project-model": "project_model",
	"resnet50": "resnet50",
	"efficientnet_b0": "efficientnet_b0",
	"efficientnet-b0": "efficientnet_b0",
}


def _normalise_backend_key(value: str | None) -> str:
	backend_key = (value or "").strip().lower()
	if not backend_key:
		return "mobilenet_v2"
	return MODEL_BACKEND_ALIASES.get(backend_key, backend_key)


MODEL_BACKEND = _normalise_backend_key(os.getenv("MODEL_BACKEND", "mobilenet_v2"))
STRICT_SOLD_CATALOG_ALIGNMENT = os.getenv("STRICT_SOLD_CATALOG_ALIGNMENT", "true").lower() == "true"
PROJECT_MODEL_CONFIDENCE_THRESHOLD = float(os.getenv("PROJECT_MODEL_CONFIDENCE_THRESHOLD", "0.80"))
PROJECT_MODEL_MARGIN_THRESHOLD = float(os.getenv("PROJECT_MODEL_MARGIN_THRESHOLD", "0.15"))

SOLD_MODEL_LABELS = (
	"kudu",
	"springbok",
	"giraffe",
	"buffalo",
	"rhino",
	"zebra",
	"ostrich",
	"elephant",
	"lion",
	"hippopotamus",
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

NOT_SOLD_CLASS_ALIASES = {
	"background",
	"not listed",
	"not sold",
	"other",
	"other animal",
	"outside catalog",
	"outside sale list",
	"unknown",
}

EXPECTED_SOLD_ANIMAL_IDS = frozenset(animal.animal_id for animal in RANCH_ANIMALS)


@dataclass(frozen=True)
class RuntimePrediction:
	animal_id: str | None
	confidence: float
	mode_label: str
	note: str
	reasons: tuple[str, ...]
	raw_label: str | None = None
	detected_species: str | None = None


@dataclass(frozen=True)
class BackendResources:
	backend_key: str
	display_name: str
	target_size: tuple[int, int]
	model: object
	preprocess_input: object
	decode_predictions: object


@dataclass(frozen=True)
class ProjectModelMetadata:
	best_model: str
	class_names: tuple[str, ...]
	display_name: str
	model_path: Path


@dataclass(frozen=True)
class ProjectModelResources:
	metadata: ProjectModelMetadata
	target_size: tuple[int, int]
	model: object


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


def is_not_sold_label(value: str | None) -> bool:
	return normalise_animal_text(value) in NOT_SOLD_CLASS_ALIASES


def _summarise_animal_ids(animal_ids: set[str]) -> str:
	labels = [get_ranch_animal(animal_id).name for animal_id in sorted(animal_ids) if get_ranch_animal(animal_id)]
	return ", ".join(labels) if labels else "none"


def discover_project_model_path(best_model: str) -> Path | None:
	for candidate in (
		ARTIFACTS_PATH / f"{best_model}_fine_tuned.h5",
		ARTIFACTS_PATH / f"{best_model}_best.h5",
	):
		if candidate.exists():
			return candidate
	return None


@lru_cache(maxsize=1)
def load_project_model_metadata() -> ProjectModelMetadata:
	if not BEST_MODEL_INFO_PATH.exists():
		raise ModelRuntimeUnavailable(
			"No best_fine_tuned_model.json artifact was found, so the project runtime cannot determine which trained model to load."
		)

	try:
		with BEST_MODEL_INFO_PATH.open("r", encoding="utf-8") as metadata_file:
			payload = json.load(metadata_file)
	except (OSError, json.JSONDecodeError) as exc:
		raise ModelRuntimeUnavailable(
			"The best_fine_tuned_model.json artifact could not be read by the runtime."
		) from exc

	best_model = str(payload.get("best_model", "")).strip()
	class_names = tuple(str(label).strip() for label in payload.get("classes", ()) if str(label).strip())
	if not best_model or not class_names:
		raise ModelRuntimeUnavailable(
			"The best_fine_tuned_model.json artifact is missing the best model name or class list."
		)

	unresolved_classes: list[str] = []
	model_sold_ids: set[str] = set()
	for label in class_names:
		if is_not_sold_label(label):
			continue

		animal_id = infer_catalog_animal_id(label)
		if animal_id is None:
			unresolved_classes.append(label)
			continue

		model_sold_ids.add(animal_id)

	if unresolved_classes:
		unresolved_list = ", ".join(unresolved_classes)
		raise ModelRuntimeUnavailable(
			"The trained model classes do not map cleanly to the configured sale catalog: "
			f"{unresolved_list}."
		)

	if STRICT_SOLD_CATALOG_ALIGNMENT and model_sold_ids != EXPECTED_SOLD_ANIMAL_IDS:
		raise ModelRuntimeUnavailable(
			"The trained model metadata does not cover the configured ten-species sale catalog. "
			f"Model species: {_summarise_animal_ids(model_sold_ids)}."
		)

	model_path = discover_project_model_path(best_model)
	if model_path is None:
		raise ModelRuntimeUnavailable(
			f"The metadata points to {best_model}, but no matching .h5 model file was found in artifacts/."
		)

	return ProjectModelMetadata(
		best_model=best_model,
		class_names=class_names,
		display_name=f"{best_model} project model",
		model_path=model_path,
	)


def _resolve_target_size(model: object) -> tuple[int, int]:
	input_shape = getattr(model, "input_shape", None)
	if isinstance(input_shape, list):
		input_shape = input_shape[0]

	if not input_shape or len(input_shape) < 3:
		return (224, 224)

	height = int(input_shape[1]) if input_shape[1] else 224
	width = int(input_shape[2]) if input_shape[2] else 224
	return (height, width)


@lru_cache(maxsize=1)
def load_project_model_resources() -> ProjectModelResources:
	try:
		import tensorflow as tf
	except Exception as exc:
		raise ModelRuntimeUnavailable(
			"TensorFlow is not installed in the current environment, so the trained project model cannot be loaded."
		) from exc

	metadata = load_project_model_metadata()

	try:
		model = tf.keras.models.load_model(metadata.model_path)
	except Exception as exc:
		raise ModelRuntimeUnavailable(
			f"The trained model file at {metadata.model_path.name} could not be loaded."
		) from exc

	return ProjectModelResources(
		metadata=metadata,
		target_size=_resolve_target_size(model),
		model=model,
	)


def _prepare_project_image_tensor(image_bytes: bytes, target_size: tuple[int, int]):
	import tensorflow as tf  # noqa: PLC0415

	image_tensor = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
	image_tensor = tf.image.resize(image_tensor, target_size)
	image_tensor = tf.cast(image_tensor, tf.float32) / 255.0
	return tf.expand_dims(image_tensor, axis=0)


def _predict_with_project_model(image_bytes: bytes) -> RuntimePrediction:
	try:
		resources = load_project_model_resources()
		image_batch = _prepare_project_image_tensor(image_bytes, resources.target_size)
		probabilities = resources.model.predict(image_batch, verbose=0)
	except ModelRuntimeUnavailable:
		raise
	except Exception as exc:
		raise ModelRuntimeUnavailable("The upload could not be processed by the trained project model.") from exc

	if len(probabilities) != 1:
		raise ModelRuntimeUnavailable("The trained project model returned an unexpected batch shape.")

	class_scores = [float(score) for score in probabilities[0]]
	if len(class_scores) != len(resources.metadata.class_names):
		raise ModelRuntimeUnavailable(
			"The trained project model output size does not match the class metadata in best_fine_tuned_model.json."
		)

	ranked_indices = sorted(range(len(class_scores)), key=class_scores.__getitem__, reverse=True)
	top_index = ranked_indices[0]
	top_label = resources.metadata.class_names[top_index]
	top_score = class_scores[top_index]
	second_score = class_scores[ranked_indices[1]] if len(ranked_indices) > 1 else 0.0
	score_margin = top_score - second_score
	top_display_label = format_species_label(top_label) or top_label

	reasons = [
		f"Prediction backend: {resources.metadata.display_name}.",
		f"Top class: {top_display_label} ({top_score:.0%}).",
	]
	if len(ranked_indices) > 1:
		second_label = resources.metadata.class_names[ranked_indices[1]]
		second_display_label = format_species_label(second_label) or second_label
		reasons.append(f"Runner-up class: {second_display_label} ({second_score:.0%}).")

	if is_not_sold_label(top_label):
		reasons.append("The trained model explicitly routed this upload into the not-sold class.")
		return RuntimePrediction(
			animal_id=None,
			confidence=max(0.51, top_score),
			mode_label=resources.metadata.display_name,
			note="The trained project model marked this upload as outside the configured sold list.",
			reasons=tuple(reasons),
			raw_label=top_label,
		)

	animal_id = infer_catalog_animal_id(top_label)
	if animal_id is None:
		reasons.append("The trained model predicted a label that does not map to any configured sold animal.")
		return RuntimePrediction(
			animal_id=None,
			confidence=max(0.51, top_score),
			mode_label=resources.metadata.display_name,
			note="The trained project model returned a class outside the configured sold-animal catalog.",
			reasons=tuple(reasons),
			raw_label=top_label,
			detected_species=top_display_label,
		)

	if top_score < PROJECT_MODEL_CONFIDENCE_THRESHOLD or score_margin < PROJECT_MODEL_MARGIN_THRESHOLD:
		reasons.append(
			"The trained model did not clear the sold-decision thresholds, so the upload was routed to not sold."
		)
		return RuntimePrediction(
			animal_id=None,
			confidence=max(0.51, top_score),
			mode_label=resources.metadata.display_name,
			note=(
				"The trained project model only marks an upload as sold when the top class is clear enough to separate it "
				"from the rest of the catalog."
			),
			reasons=tuple(reasons),
			raw_label=top_label,
		)

	animal = get_ranch_animal(animal_id)
	reasons.append("The trained model matched a configured sold species and cleared the sold-decision thresholds.")
	return RuntimePrediction(
		animal_id=animal_id,
		confidence=max(0.52, top_score),
		mode_label=resources.metadata.display_name,
		note="This result came from the fine-tuned project model selected by best_fine_tuned_model.json.",
		reasons=tuple(reasons),
		raw_label=top_label,
		detected_species=animal.name if animal is not None else top_display_label,
	)


@lru_cache(maxsize=1)
def load_imagenet_backend_resources() -> BackendResources:
	try:
		import tensorflow as tf
	except Exception as exc:
		raise ModelRuntimeUnavailable(
			"TensorFlow is not installed in the current environment, so the ImageNet runtime is unavailable."
		) from exc

	backend_key = MODEL_BACKEND

	try:
		if backend_key == "mobilenet_v2":
			from keras.applications.mobilenet_v2 import (  # noqa: PLC0415
				MobileNetV2,
				decode_predictions,
				preprocess_input,
			)

			return BackendResources(
				backend_key=backend_key,
				display_name="MobileNetV2 runtime",
				target_size=(224, 224),
				model=MobileNetV2(weights="imagenet"),
				preprocess_input=preprocess_input,
				decode_predictions=decode_predictions,
			)

		if backend_key == "mobilenet_v3_small":
			from keras.applications.mobilenet_v3 import (  # noqa: PLC0415
				MobileNetV3Small,
				decode_predictions,
				preprocess_input,
			)

			return BackendResources(
				backend_key=backend_key,
				display_name="MobileNetV3 Small runtime",
				target_size=(224, 224),
				model=MobileNetV3Small(weights="imagenet"),
				preprocess_input=preprocess_input,
				decode_predictions=decode_predictions,
			)

		if backend_key == "efficientnet_b0":
			from keras.applications.efficientnet import (  # noqa: PLC0415
				EfficientNetB0,
				decode_predictions,
				preprocess_input,
			)

			return BackendResources(
				backend_key=backend_key,
				display_name="EfficientNetB0 runtime",
				target_size=(224, 224),
				model=EfficientNetB0(weights="imagenet"),
				preprocess_input=preprocess_input,
				decode_predictions=decode_predictions,
			)

		if backend_key == "resnet50":
			from keras.applications.resnet50 import (  # noqa: PLC0415
				ResNet50,
				decode_predictions,
				preprocess_input,
			)

			return BackendResources(
				backend_key=backend_key,
				display_name="ResNet50 runtime",
				target_size=(224, 224),
				model=ResNet50(weights="imagenet"),
				preprocess_input=preprocess_input,
				decode_predictions=decode_predictions,
			)
	except Exception as exc:
		raise ModelRuntimeUnavailable(
			f"The {backend_key} backend could not be loaded. Install TensorFlow and the model weights or switch MODEL_BACKEND."
		) from exc

	raise ModelRuntimeUnavailable(
		"Unsupported MODEL_BACKEND. Use mobilenet, mobilenet_v2, mobilenet_v3_small, project_model, efficientnet_b0, or resnet50."
	)


def _prepare_imagenet_image_tensor(image_bytes: bytes, target_size: tuple[int, int]):
	import tensorflow as tf  # noqa: PLC0415

	image_tensor = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
	image_tensor = tf.image.resize(image_tensor, target_size)
	image_tensor = tf.cast(image_tensor, tf.float32)
	return tf.expand_dims(image_tensor, axis=0)


def _decode_imagenet_predictions(image_bytes: bytes) -> tuple[BackendResources, list[tuple[str, str, float]]]:
	resources = load_imagenet_backend_resources()
	image_batch = _prepare_imagenet_image_tensor(image_bytes, resources.target_size)
	processed_batch = resources.preprocess_input(image_batch)
	predictions = resources.model.predict(processed_batch, verbose=0)
	decoded_predictions = resources.decode_predictions(predictions, top=5)[0]
	return resources, [(class_id, label, float(score)) for class_id, label, score in decoded_predictions]


def _predict_with_imagenet_backend(image_bytes: bytes, file_name: str | None = None) -> RuntimePrediction:
	try:
		resources, decoded_predictions = _decode_imagenet_predictions(image_bytes)
	except ModelRuntimeUnavailable:
		raise
	except Exception as exc:
		raise ModelRuntimeUnavailable("The image could not be processed by the selected ImageNet runtime.") from exc

	if not decoded_predictions:
		raise ModelRuntimeUnavailable("The selected ImageNet backend returned no predictions.")

	top_class_id, top_label, top_score = decoded_predictions[0]
	top_display_label = format_species_label(top_label) or top_label
	reasons = [
		f"Prediction backend: {resources.display_name}.",
		f"Top label: {top_display_label} ({top_score:.0%}).",
	]

	resolved_sold_label = resolve_sold_model_label(top_label)
	if resolved_sold_label:
		animal_id = next(
			catalog_id
			for catalog_id, label in CATALOG_ID_TO_MODEL_LABEL.items()
			if label == resolved_sold_label
		)
		reasons.append("The top ImageNet label maps directly to one of the configured sold-animal classes.")
		return RuntimePrediction(
			animal_id=animal_id,
			confidence=max(0.52, top_score),
			mode_label=resources.display_name,
			note=(
				"This runtime uses a generic ImageNet backbone plus a strict sold-animal label map. "
				"Use the project_model backend if you want the app to follow train_models.py artifacts directly."
			),
			reasons=tuple(reasons),
			raw_label=top_class_id,
			detected_species=top_display_label,
		)

	file_name_match = infer_catalog_animal(file_name)
	if file_name_match is not None:
		reasons.append(
			"The visual model output did not land on a sold class, but the upload name matched one of the configured sold-animal aliases."
		)
		return RuntimePrediction(
			animal_id=file_name_match.animal_id,
			confidence=0.68,
			mode_label="Filename-assisted fallback",
			note=(
				"The visual model did not resolve cleanly to one of the sold classes, so the temporary site fell back to the upload name."
			),
			reasons=tuple(reasons),
			raw_label=top_class_id,
			detected_species=file_name_match.name,
		)

	detected_species = infer_outside_species_name(file_name) or infer_outside_species_name(top_label) or top_display_label
	reasons.append("The top runtime label did not map to any configured sold-animal class, so the upload is treated as not sold.")
	return RuntimePrediction(
		animal_id=None,
		confidence=max(0.51, top_score),
		mode_label=resources.display_name,
		note=(
			"Anything outside the configured kudu, springbok, giraffe, buffalo, rhino, zebra, ostrich, elephant, lion, and hippopotamus list is routed to not sold."
		),
		reasons=tuple(reasons),
		raw_label=top_class_id,
		detected_species=detected_species,
	)


def predict_with_runtime(image_bytes: bytes, file_name: str | None = None) -> RuntimePrediction:
	if MODEL_BACKEND == "project_model":
		return _predict_with_project_model(image_bytes)

	return _predict_with_imagenet_backend(image_bytes=image_bytes, file_name=file_name)