from __future__ import annotations

import base64
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlencode

from dash import Dash, Input, Output, State, dcc, html


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
	create_home_page,
	create_layout,
	error_preview,
	error_result_card,
	image_preview,
	result_card,
)
from app.animal_location_data import get_animal_location, get_animal_locations  # noqa: E402
from app.animal_range_map import (  # noqa: E402
	build_animal_location_map,
	build_district_detail_map,
	build_sa_range_map,
	get_species_range_context,
	resolve_animal_location,
	resolve_animal_locations,
)

USE_TRAINED_MODEL = os.getenv("USE_TRAINED_MODEL", "false").lower() == "true"
SUPPORTED_IMAGE_TYPES = {
	"image/jpeg": "JPG",
	"image/png": "PNG",
	"image/webp": "WEBP",
}
RANCH_ANIMAL_HINTS = {
	"kudu": "greater-kudu",
	"springbok": "springbok",
	"zebra": "plains-zebra",
	"giraffe": "cape-giraffe",
	"wildebeest": "blue-wildebeest",
	"gnu": "blue-wildebeest",
	"ostrich": "common-ostrich",
}
OUTSIDE_RANCH_SPECIES = {
	"lion": "lion",
	"leopard": "leopard",
	"cheetah": "cheetah",
	"rhino": "rhinoceros",
	"rhinoceros": "rhinoceros",
	"hippo": "hippopotamus",
	"hippopotamus": "hippopotamus",
	"elephant": "elephant",
	"buffalo": "buffalo",
	"warthog": "warthog",
	"crocodile": "crocodile",
	"penguin": "penguin",
	"whale": "whale",
	"shark": "shark",
}
OUTSIDE_RANCH_HINTS = tuple(OUTSIDE_RANCH_SPECIES)


@dataclass(frozen=True)
class RanchAnimal:
	animal_id: str
	name: str
	scientific_name: str
	category: str
	short_description: str
	description: str
	habitat_zone: str
	best_viewing: str
	image_src: str
	traits: tuple[str, ...]

	@property
	def page_href(self) -> str:
		return "/animals?" + urlencode({"animal": self.animal_id})


RANCH_ANIMALS = (
	RanchAnimal(
		"greater-kudu",
		"Greater Kudu",
		"Tragelaphus strepsiceros",
		"Antelope",
		"A tall spiral-horned browser that moves quietly through thornveld and acacia cover.",
		"The greater kudu is one of the ranch's signature antelope species. Bulls carry long spiral horns while cows and young animals move in looser groups through thicker bush. Visitors usually spot them near the shaded thornveld edge where they browse on leaves and pods rather than open grass.",
		"Acacia ridge and mixed bushveld line",
		"Early morning near shaded browsing paths",
		"/assets/assets/greater-kudu.jpg",
		(
			"Look for the white body stripes and the dark ridge running along the spine.",
			"Mature bulls carry dramatic spiral horns that stand out against the skyline.",
			"They tend to pause in cover before stepping into open clearings.",
		),
	),
	RanchAnimal(
		"springbok",
		"Springbok",
		"Antidorcas marsupialis",
		"Antelope",
		"A light, fast antelope known for its high pronking leap across the open veld.",
		"Springbok are among the most recognisable South African antelope. On the ranch they are most often seen out in the open, grazing in loose groups and suddenly springing into the air when startled. Their white face markings and chestnut side band make them easy to identify in bright daylight.",
		"Open grass flats and short-grazing camps",
		"Late afternoon across the sunlit plains",
		"/assets/assets/springbok.jpg",
		(
			"Watch for the chestnut stripe along each side of the body.",
			"Pronking jumps are a common sign that the herd is alert.",
			"They usually prefer the more open grazing areas of the ranch.",
		),
	),
	RanchAnimal(
		"plains-zebra",
		"Plains Zebra",
		"Equus quagga",
		"Grazers",
		"A bold black-and-cream grazer that moves in family groups across the plains.",
		"Plains zebra bring strong movement and contrast to the ranch landscape. Their striping is unique to each animal, but the herd behavior is just as helpful when identifying them: they graze together, watch in different directions, and often share open ground with antelope species.",
		"Central grazing paddocks and waterline paths",
		"Mid-morning around open water points",
		"/assets/assets/plains-zebra.jpg",
		(
			"Each zebra has its own stripe pattern even though the herd appears uniform from a distance.",
			"They often stand in loose lines while grazing or resting.",
			"Their rounded ears and upright mane help distinguish them from antelope at range.",
		),
	),
	RanchAnimal(
		"cape-giraffe",
		"Cape Giraffe",
		"Giraffa giraffa giraffa",
		"Browsers",
		"A towering browser that strips leaves from the upper acacia canopy.",
		"Cape giraffe are among the easiest animals to spot on the ranch because they rise above the thorn trees. They browse quietly for long periods and often move with measured steps between tree lines. Their pale patches and long silhouette make them one of the most popular animals in the catalog.",
		"Tall acacia belt on the northern loop",
		"Early afternoon where tree canopies stay green",
		"/assets/assets/cape-giraffe.jpg",
		(
			"Notice the patch pattern and the dark tail tuft moving while they feed.",
			"They often browse at heights other animals cannot reach.",
			"Groups may include calves staying close to the adults on the edges of the trees.",
		),
	),
	RanchAnimal(
		"blue-wildebeest",
		"Blue Wildebeest",
		"Connochaetes taurinus",
		"Plains herd",
		"A heavy-shouldered herd animal with a dark mane and curved horns.",
		"Blue wildebeest add weight and motion to the ranch's open grassland scenes. Their broad front build and darker face stand out clearly when they gather in small groups. They often keep moving between open grazing lines and water access points throughout the day.",
		"Southern plains and waterline route",
		"Morning and late afternoon near grazing lines",
		"/assets/assets/blue-wildebeest.jpg",
		(
			"Look for the beard, dark mane, and sloping back line.",
			"They often travel in short bursts before stopping together.",
			"The horn shape is broader and heavier than that of antelope species.",
		),
	),
	RanchAnimal(
		"common-ostrich",
		"Common Ostrich",
		"Struthio camelus",
		"Birdlife",
		"The world's largest bird, built for speed across dry South African ground.",
		"Ostriches bring a very different silhouette to the ranch catalog. Their height, bare legs, and long neck make them easy to identify even from a distance. They are usually seen moving across open areas with quick, direct strides and pausing to scan the surroundings.",
		"Dry grass camps and open fence lines",
		"Warm daylight hours on open ground",
		"/assets/assets/common-ostrich.jpg",
		(
			"The long neck and powerful legs are visible even from far away.",
			"They tend to prefer open areas with a clear line of sight.",
			"Fast running is one of the clearest behavioral cues when they are disturbed.",
		),
	),
)
RANCH_ANIMALS_BY_ID = {animal.animal_id: animal for animal in RANCH_ANIMALS}


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
	range_map_figure: object | None = None
	action_href: str | None = None
	action_label: str | None = None
	secondary_action_href: str | None = None
	secondary_action_label: str | None = None


app = Dash(
	__name__,
	title="Bosvelder Ranch Catalog",
	assets_folder=str(ASSETS_PATH),
	suppress_callback_exceptions=True,
	update_title=None,
)
server = app.server


def get_ranch_animal(animal_id: str | None) -> RanchAnimal | None:
	if not animal_id:
		return None
	return RANCH_ANIMALS_BY_ID.get(animal_id)


def get_animal_location_context(animal: RanchAnimal) -> tuple[tuple, tuple, object | None]:
	locations = get_animal_locations(animal.animal_id)
	resolved_locations = resolve_animal_locations(locations)
	location_map_figure = build_animal_location_map(animal.name, locations)
	if location_map_figure is None:
		location_map_figure = build_sa_range_map(animal.name)
	return locations, resolved_locations, location_map_figure


def select_ranch_animal(file_name: str, image_bytes: bytes) -> RanchAnimal:
	lowered_name = file_name.lower()
	for keyword, animal_id in RANCH_ANIMAL_HINTS.items():
		if keyword in lowered_name:
			return RANCH_ANIMALS_BY_ID[animal_id]

	index = int(hashlib.sha256(image_bytes).hexdigest()[8:10], 16) % len(RANCH_ANIMALS)
	return RANCH_ANIMALS[index]


def infer_species_name(file_name: str) -> str | None:
	lowered_name = file_name.lower()
	for keyword, animal_id in RANCH_ANIMAL_HINTS.items():
		if keyword in lowered_name:
			return RANCH_ANIMALS_BY_ID[animal_id].name

	for keyword, species_name in OUTSIDE_RANCH_SPECIES.items():
		if keyword in lowered_name:
			return species_name

	return None


def get_range_map_payload(species_name: str | None) -> tuple[str | None, tuple[str, ...], object | None]:
	range_context = get_species_range_context(species_name)
	if range_context is None:
		return None, (), None

	return (
		(
			f"Highlighted provinces show the South African areas currently linked to {range_context.display_name} "
			"in this app's reference map."
		),
		range_context.provinces,
		build_sa_range_map(range_context.canonical_name),
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


def run_model_prediction(image_bytes: bytes, file_name: str) -> PredictionResult:
	raise NotImplementedError("Connect your trained ranch animal classifier here.")


def run_demo_prediction(
	file_name: str,
	mime_type: str,
	image_bytes: bytes,
	mode_label: str,
) -> PredictionResult:
	lowered_name = file_name.lower()
	detected_species = infer_species_name(file_name)
	digest_score = int(hashlib.sha256(image_bytes).hexdigest()[:8], 16) / 0xFFFFFFFF
	positive_hits = sum(keyword in lowered_name for keyword in RANCH_ANIMAL_HINTS)
	negative_hits = sum(keyword in lowered_name for keyword in OUTSIDE_RANCH_HINTS)
	adjusted_score = digest_score + (positive_hits * 0.22) - (negative_hits * 0.25)
	adjusted_score = max(0.03, min(0.97, adjusted_score))
	match_found = adjusted_score >= 0.5
	confidence = 0.6 + min(abs(adjusted_score - 0.5) * 0.65, 0.34)

	reasons: list[str] = []
	if positive_hits:
		reasons.append("The upload name contains animal keywords that match the Bosvelder Ranch species list.")
	if negative_hits:
		reasons.append("The upload name contains species hints that currently sit outside the Bosvelder Ranch catalog.")
	reasons.append(
		"This build still uses deterministic demo matching, so identical uploads return the same placeholder result until the trained model is connected."
	)
	reasons.append(
		f"Detected file type: {SUPPORTED_IMAGE_TYPES[mime_type]}. Swap this rule set for your real classifier when the model artifact is ready."
	)

	if match_found:
		animal = select_ranch_animal(file_name, image_bytes)
		range_summary, range_provinces, range_map_figure = get_range_map_payload(animal.name)
		reasons.append(f"Matched ranch animal route: {animal.name}.")
		return PredictionResult(
			title=animal.name,
			badge_text="Exists on the ranch",
			confidence=confidence,
			badge_class="result-badge--positive",
			fill_class="confidence-fill--positive",
			summary=(
				"This upload matches one of the South African animals currently listed on Bosvelder Ranch. You can now open the animal map page or browse the full catalog."
			),
			reasons=tuple(reasons),
			mode_label=mode_label,
			details=(
				("Species", animal.scientific_name),
				("Habitat zone", animal.habitat_zone),
				("Best viewing", animal.best_viewing),
			),
			note="This is still a demo result. When the real model is connected, the page flow can stay the same while only the recognition step changes.",
			range_summary=range_summary,
			range_provinces=range_provinces,
			range_map_figure=range_map_figure,
			action_href=animal.page_href,
			action_label="Open animal map",
			secondary_action_href="/catalog",
			secondary_action_label="View full catalog",
		)

	range_summary = None
	range_provinces: tuple[str, ...] = ()
	range_map_figure = None
	title = "Animal not found on this ranch"
	summary = (
		"We could not confidently match this upload to the current Bosvelder Ranch animal list. Try another image or upload a clearer photo of the species."
	)
	details = (
		("Catalog scope", "Bosvelder Ranch species list"),
		("Status", "No confident match"),
		("Next step", "Try another image"),
	)

	if detected_species:
		range_context = get_species_range_context(detected_species)
		if range_context is not None:
			range_summary, range_provinces, range_map_figure = get_range_map_payload(detected_species)
			title = f"{range_context.display_name} is not part of the ranch catalog"
			summary = (
				f"This upload appears closest to {range_context.display_name} based on the current demo hints, "
				"but that species is not listed on Bosvelder Ranch. The province map below shows where it is commonly associated in South Africa."
			)
			details = (
				("Detected species", range_context.display_name),
				("Catalog status", "Not on Bosvelder Ranch"),
				("Next step", "Open the full catalog"),
			)
			reasons.append(
				f"Detected species cue: {range_context.display_name}. The current catalog treats it as outside the Bosvelder Ranch list."
			)

	return PredictionResult(
		title=title,
		badge_text="Not on the ranch",
		confidence=confidence,
		badge_class="result-badge--negative",
		fill_class="confidence-fill--negative",
		summary=summary,
		reasons=tuple(reasons),
		mode_label=mode_label,
		details=details,
		note="This negative result is also placeholder logic until the trained classifier is wired in.",
		range_summary=range_summary,
		range_provinces=range_provinces,
		range_map_figure=range_map_figure,
		secondary_action_href="/catalog",
		secondary_action_label="View full catalog",
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
				mode_label="Demo fallback",
			)

	return run_demo_prediction(
		file_name=file_name,
		mime_type=mime_type,
		image_bytes=image_bytes,
		mode_label="Demo ranch search",
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

		locations, resolved_locations, location_map_figure = get_animal_location_context(animal)
		return create_animal_page(
			animal,
			locations=locations,
			resolved_locations=resolved_locations,
			location_map_figure=location_map_figure,
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
		district_map_figure = build_district_detail_map(location)
		return create_district_page(
			animal,
			location,
			resolved_location,
			district_map_figure,
		)

	return create_home_page()
@app.callback(
	Output("image-preview", "children"),
	Output("result-panel", "children"),
	Input("image-upload", "contents"),
	State("image-upload", "filename"),
)
def update_prediction(contents: str | None, file_name: str | None):
	if not contents:
		return None, None

	safe_name = normalise_file_name(file_name)

	try:
		mime_type, image_bytes = decode_upload(contents)
		prediction = classify_upload(safe_name, mime_type, image_bytes)
	except ValueError as exc:
		message = str(exc)
		return (
			html.Section(className="panel preview-panel-shell", children=error_preview(message)),
			html.Section(className="panel result-panel-shell", children=error_result_card(message)),
		)

	size_kb = len(image_bytes) / 1024
	return (
		html.Section(
			className="panel preview-panel-shell",
			children=image_preview(contents, safe_name, size_kb),
		),
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
