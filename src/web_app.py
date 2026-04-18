from __future__ import annotations

import base64
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dash import Dash, Input, Output, State


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_PATH = PROJECT_ROOT / "app"
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app.layout import (  # noqa: E402
	create_layout,
	default_result_card,
	empty_preview,
	error_preview,
	error_result_card,
	image_preview,
	result_card,
)

USE_TRAINED_MODEL = os.getenv("USE_TRAINED_MODEL", "false").lower() == "true"
SUPPORTED_IMAGE_TYPES = {
	"image/jpeg": "JPG",
	"image/png": "PNG",
	"image/webp": "WEBP",
}
CATEGORY_HINTS = (
	"bag",
	"belt",
	"dress",
	"headphones",
	"hoodie",
	"jacket",
	"jeans",
	"laptop",
	"phone",
	"shoe",
	"shirt",
	"skirt",
	"sneaker",
	"watch",
)
OUT_OF_SCOPE_HINTS = (
	"animal",
	"car",
	"dog",
	"flower",
	"house",
	"landscape",
	"pet",
	"tree",
	"truck",
)


@dataclass(frozen=True)
class PredictionResult:
	label: str
	confidence: float
	badge_class: str
	fill_class: str
	summary: str
	reasons: tuple[str, ...]
	mode_label: str


app = Dash(
	__name__,
	title="Catalog Gate Demo",
	assets_folder=str(ASSETS_PATH),
	update_title=None,
)
server = app.server


def normalise_file_name(file_name: str | None) -> str:
	if file_name and file_name.strip():
		return file_name.strip()
	return "uploaded-image"


def decode_upload(contents: str) -> tuple[str, bytes]:
	if not contents or "," not in contents:
		raise ValueError("The upload payload is invalid. Please select the image again.")

	header, encoded = contents.split(",", maxsplit=1)
	mime_type = header.removeprefix("data:").removesuffix(";base64")

	if mime_type not in SUPPORTED_IMAGE_TYPES:
		raise ValueError("Only JPG, PNG, and WEBP images are supported in this prototype.")

	try:
		image_bytes = base64.b64decode(encoded, validate=True)
	except ValueError as exc:
		raise ValueError("The uploaded image could not be decoded.") from exc

	if not image_bytes:
		raise ValueError("The uploaded image is empty.")

	return mime_type, image_bytes


def run_model_prediction(image_bytes: bytes, file_name: str) -> PredictionResult:
	raise NotImplementedError("Connect your trained model inference here.")


def run_demo_prediction(
	file_name: str,
	mime_type: str,
	image_bytes: bytes,
	mode_label: str,
) -> PredictionResult:
	lowered_name = file_name.lower()
	digest_score = int(hashlib.sha256(image_bytes).hexdigest()[:8], 16) / 0xFFFFFFFF
	positive_hits = sum(keyword in lowered_name for keyword in CATEGORY_HINTS)
	negative_hits = sum(keyword in lowered_name for keyword in OUT_OF_SCOPE_HINTS)
	adjusted_score = digest_score + (positive_hits * 0.18) - (negative_hits * 0.18)
	adjusted_score = max(0.04, min(0.96, adjusted_score))
	catalog_match = adjusted_score >= 0.5
	confidence = 0.58 + min(abs(adjusted_score - 0.5) * 0.7, 0.35)

	reasons: list[str] = []
	if positive_hits:
		reasons.append("The file name contains product keywords that fit a store catalog.")
	if negative_hits:
		reasons.append("The file name contains keywords that usually sit outside an e-commerce catalog.")
	reasons.append(
		"The decision is deterministic, using the upload fingerprint, so the same image returns the same demo result."
	)
	reasons.append(
		f"Detected file type: {SUPPORTED_IMAGE_TYPES[mime_type]}. Replace this demo rule set once the model artifact is ready."
	)

	if catalog_match:
		return PredictionResult(
			label="Sold by company",
			confidence=confidence,
			badge_class="result-badge--positive",
			fill_class="confidence-fill--positive",
			summary=(
				"This upload is currently routed into the in-catalog product flow. "
				"Use it to demonstrate the expected approval experience."
			),
			reasons=tuple(reasons),
			mode_label=mode_label,
		)

	return PredictionResult(
		label="Not sold by company",
		confidence=confidence,
		badge_class="result-badge--negative",
		fill_class="confidence-fill--negative",
		summary=(
			"This upload is currently routed into the out-of-catalog flow. "
			"Use it to demonstrate the rejection experience before training is done."
		),
		reasons=tuple(reasons),
		mode_label=mode_label,
	)


def classify_upload(file_name: str, mime_type: str, image_bytes: bytes) -> PredictionResult:
	if USE_TRAINED_MODEL:
		try:
			return run_model_prediction(image_bytes=image_bytes, file_name=file_name)
		except NotImplementedError:
			return run_demo_prediction(
				file_name=file_name,
				mime_type=mime_type,
				image_bytes=image_bytes,
				mode_label="Prototype fallback",
			)

	return run_demo_prediction(
		file_name=file_name,
		mime_type=mime_type,
		image_bytes=image_bytes,
		mode_label="Prototype mode",
	)


app.layout = create_layout()


@app.callback(
	Output("image-preview", "children"),
	Output("result-panel", "children"),
	Input("image-upload", "contents"),
	State("image-upload", "filename"),
)
def update_prediction(contents: str | None, file_name: str | None):
	if not contents:
		return empty_preview(), default_result_card()

	safe_name = normalise_file_name(file_name)

	try:
		mime_type, image_bytes = decode_upload(contents)
		prediction = classify_upload(safe_name, mime_type, image_bytes)
	except ValueError as exc:
		message = str(exc)
		return error_preview(message), error_result_card(message)

	size_kb = len(image_bytes) / 1024
	return (
		image_preview(contents, safe_name, size_kb),
		result_card(prediction, safe_name, SUPPORTED_IMAGE_TYPES[mime_type], size_kb),
	)


if __name__ == "__main__":
	app.run(
		host="0.0.0.0",
		port=int(os.getenv("PORT", "8050")),
		debug=os.getenv("DASH_DEBUG", "false").lower() == "true",
	)
