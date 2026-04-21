from __future__ import annotations

import base64
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs
from uuid import uuid4

from dash import Dash, Input, Output, State, dcc, html, no_update


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_PATH = PROJECT_ROOT / "app"
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app.layout import (  # noqa: E402
	create_animal_missing_page,
	create_animal_page,
	create_catalog_page,
	create_district_missing_page,
	create_district_page,
	build_upload_component,
	create_home_page,
	create_layout,
	error_result_card,
	result_card,
)
from app.animal_catalog import (  # noqa: E402
	RANCH_ANIMALS,
	RanchAnimal,
	get_ranch_animal,
	infer_catalog_animal,
)
from app.animal_location_data import get_animal_location, get_animal_locations  # noqa: E402
from app.animal_range_map import (  # noqa: E402
	get_animal_location_map_asset_src,
	get_district_map_asset_src,
	get_range_map_asset_src,
	get_species_range_context,
	resolve_animal_location,
	resolve_animal_locations,
)
from src.model_runtime import (  # noqa: E402
	ModelRuntimeUnavailable,
	infer_outside_species_name,
	predict_with_runtime,
)

USE_TRAINED_MODEL = os.getenv("USE_TRAINED_MODEL", "true").lower() == "true"
SUPPORTED_IMAGE_TYPES = {
	"image/jpeg": "JPG",
	"image/png": "PNG",
	"image/webp": "WEBP",
}


@dataclass(frozen=True)
class PredictionResult:
	title: str
	badge_text: str
	confidence: float
	badge_class: str
	fill_class: str
	summary: str
	reasons: tuple[str, ...]
	mode_label: str
	details: tuple[tuple[str, str], ...]
	note: str
	range_summary: str | None = None
	range_provinces: tuple[str, ...] = ()
	range_map_image_src: str | None = None
	action_href: str | None = None
	action_label: str | None = None
	secondary_action_href: str | None = None
	secondary_action_label: str | None = None


app = Dash(
	__name__,
	title="Bosvelder Sale Catalog",
	assets_folder=str(ASSETS_PATH),
	suppress_callback_exceptions=True,
	update_title=None,
)
server = app.server


def get_animal_location_context(animal: RanchAnimal) -> tuple[tuple, tuple, str | None]:
	locations = get_animal_locations(animal.animal_id)
	resolved_locations = resolve_animal_locations(locations)
	location_map_image_src = get_animal_location_map_asset_src(animal.animal_id)
	return locations, resolved_locations, location_map_image_src


def get_range_map_payload(species_name: str | None) -> tuple[str | None, tuple[str, ...], str | None]:
	range_context = get_species_range_context(species_name)
	if range_context is None:
		return None, (), None

	return (
		(
			f"Highlighted provinces show the South African areas currently linked to {range_context.display_name} "
			"in this app's reference map."
		),
		range_context.provinces,
		get_range_map_asset_src(range_context.canonical_name),
	)


def normalise_file_name(file_name: str | None) -> str:
	if file_name and file_name.strip():
		return file_name.strip()
	return "uploaded-animal"


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


def build_sold_prediction(
	animal: RanchAnimal,
	confidence: float,
	mode_label: str,
	reasons: list[str],
	note: str,
) -> PredictionResult:
	range_summary, range_provinces, range_map_image_src = get_range_map_payload(animal.name)
	return PredictionResult(
		title=animal.name,
		badge_text="Sold",
		confidence=confidence,
		badge_class="result-badge--positive",
		fill_class="confidence-fill--positive",
		summary=(
			"This upload matches one of the ten animals currently listed for sale. "
			"You can open the animal page or browse the full sale catalog."
		),
		reasons=tuple(reasons),
		mode_label=mode_label,
		details=(
			("Species", animal.scientific_name),
			("Habitat zone", animal.habitat_zone),
			("Catalog status", "Sold"),
		),
		note=note,
		range_summary=range_summary,
		range_provinces=range_provinces,
		range_map_image_src=range_map_image_src,
		action_href=animal.page_href,
		action_label="Open animal page",
		secondary_action_href="/catalog",
		secondary_action_label="View full catalog",
	)


def build_not_sold_prediction(
	detected_species: str | None,
	confidence: float,
	mode_label: str,
	reasons: list[str],
	note: str,
) -> PredictionResult:
	range_summary = None
	range_provinces: tuple[str, ...] = ()
	range_map_image_src = None
	title = "Not sold"
	summary = (
		"This upload does not match one of the ten animals currently listed for sale. "
		"Anything outside that configured list is routed to the not sold category."
	)
	details = (
		("Detected species", detected_species or "Unresolved"),
		("Catalog status", "Not sold"),
		("Next step", "Upload another image"),
	)

	if detected_species:
		title = f"{detected_species} is not sold"
		range_summary, range_provinces, range_map_image_src = get_range_map_payload(detected_species)
		if range_summary is not None:
			summary = (
				f"This upload looks closest to {detected_species}, but that species is outside the current sold list. "
				"The province map below shows where it is commonly associated in South Africa."
			)

	return PredictionResult(
		title=title,
		badge_text="Not sold",
		confidence=confidence,
		badge_class="result-badge--negative",
		fill_class="confidence-fill--negative",
		summary=summary,
		reasons=tuple(reasons),
		mode_label=mode_label,
		details=details,
		note=note,
		range_summary=range_summary,
		range_provinces=range_provinces,
		range_map_image_src=range_map_image_src,
		secondary_action_href="/catalog",
		secondary_action_label="View full catalog",
	)


def run_model_prediction(image_bytes: bytes, file_name: str) -> PredictionResult:
	runtime_prediction = predict_with_runtime(image_bytes=image_bytes, file_name=file_name)
	reasons = list(runtime_prediction.reasons)

	if runtime_prediction.animal_id:
		animal = get_ranch_animal(runtime_prediction.animal_id)
		if animal is not None:
			reasons.append(f"Resolved sold listing: {animal.name}.")
			return build_sold_prediction(
				animal=animal,
				confidence=runtime_prediction.confidence,
				mode_label=runtime_prediction.mode_label,
				reasons=reasons,
				note=runtime_prediction.note,
			)

	reasons.append("The runtime output did not resolve to a configured catalog entry, so the upload was routed to not sold.")
	return build_not_sold_prediction(
		detected_species=runtime_prediction.detected_species,
		confidence=runtime_prediction.confidence,
		mode_label=runtime_prediction.mode_label,
		reasons=reasons,
		note=runtime_prediction.note,
	)


def run_demo_prediction(
	file_name: str,
	mime_type: str,
	mode_label: str,
	runtime_note: str | None = None,
) -> PredictionResult:
	matched_animal = infer_catalog_animal(file_name)
	detected_species = infer_outside_species_name(file_name)
	reasons = [
		f"Detected file type: {SUPPORTED_IMAGE_TYPES[mime_type]}.",
		"The temporary site is using deterministic filename matching because the live model runtime is disabled or unavailable.",
	]
	if runtime_note:
		reasons.append(f"Trained runtime unavailable: {runtime_note}")

	if matched_animal is not None:
		reasons.append("The upload name matched one of the exact sold-animal aliases configured for this project.")
		return build_sold_prediction(
			animal=matched_animal,
			confidence=0.86,
			mode_label=mode_label,
			reasons=reasons,
			note=(
				"This is deterministic fallback logic for the temporary site. Enable the runtime backend if you want the upload itself, rather than the file name, to drive the result."
			),
		)

	if detected_species:
		reasons.append(f"Detected not-sold species cue: {detected_species}.")
	else:
		reasons.append("No sold-animal alias was found in the upload name, so demo mode routes the image to not sold.")

	return build_not_sold_prediction(
		detected_species=detected_species,
		confidence=0.62 if detected_species else 0.55,
		mode_label=mode_label,
		reasons=reasons,
		note=(
			"This fallback is intentionally strict: only the configured kudu, springbok, giraffe, buffalo, rhino, zebra, ostrich, elephant, lion, and hippopotamus aliases resolve to sold."
		),
	)


def classify_upload(file_name: str, mime_type: str, image_bytes: bytes) -> PredictionResult:
	if USE_TRAINED_MODEL:
		try:
			return run_model_prediction(image_bytes=image_bytes, file_name=file_name)
		except ModelRuntimeUnavailable as exc:
			return run_demo_prediction(
				file_name=file_name,
				mime_type=mime_type,
				mode_label="Demo fallback",
				runtime_note=str(exc),
			)

	return run_demo_prediction(
		file_name=file_name,
		mime_type=mime_type,
		mode_label="Deterministic temp match",
	)


app.layout = create_layout()
app.validation_layout = html.Div(
	[
		create_layout(),
		create_home_page(),
		create_catalog_page(RANCH_ANIMALS),
		create_animal_page(RANCH_ANIMALS[0]),
		create_animal_missing_page(),
		create_district_missing_page(),
	]
)


@app.callback(
	Output("page-content", "children"),
	Input("url", "pathname"),
	Input("url", "search"),
)
def render_page(pathname: str | None, search: str | None):
	query_params = parse_qs((search or "").lstrip("?"))

	if pathname == "/catalog":
		return create_catalog_page(RANCH_ANIMALS)

	if pathname == "/animals":
		animal = get_ranch_animal(query_params.get("animal", [None])[0])
		if animal is None:
			return create_animal_missing_page()

		locations, resolved_locations, location_map_image_src = get_animal_location_context(animal)
		return create_animal_page(
			animal,
			locations=locations,
			resolved_locations=resolved_locations,
			location_map_image_src=location_map_image_src,
		)

	if pathname == "/districts":
		animal = get_ranch_animal(query_params.get("animal", [None])[0])
		location = get_animal_location(
			query_params.get("animal", [None])[0],
			query_params.get("location", [None])[0],
		)
		if animal is None or location is None:
			return create_district_missing_page()

		resolved_location = resolve_animal_location(location)
		district_map_image_src = get_district_map_asset_src(animal.animal_id, location.location_id)
		return create_district_page(
			animal,
			location,
			resolved_location,
			district_map_image_src,
		)

	return create_home_page()
@app.callback(
	Output("upload-shell", "children"),
	Output("result-panel", "children"),
	Input("image-upload", "contents"),
	State("image-upload", "filename"),
)
def update_prediction(contents: str | None, file_name: str | None):
	if not contents:
		return no_update, None

	safe_name = normalise_file_name(file_name)
	refreshed_upload = build_upload_component(uuid4().hex)

	try:
		mime_type, image_bytes = decode_upload(contents)
		prediction = classify_upload(safe_name, mime_type, image_bytes)
	except ValueError as exc:
		message = str(exc)
		return (
			refreshed_upload,
			html.Section(className="panel result-panel-shell", children=error_result_card(message)),
		)

	return (
		refreshed_upload,
		html.Section(
			className="panel result-panel-shell",
			children=result_card(prediction),
		),
	)


if __name__ == "__main__":
	app.run(
		host="0.0.0.0",
		port=int(os.getenv("PORT", "8050")),
		debug=os.getenv("DASH_DEBUG", "false").lower() == "true",
	)
