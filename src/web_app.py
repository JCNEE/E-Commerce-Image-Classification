from __future__ import annotations

import base64
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlencode

from dash import Dash, Input, Output, State, html


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_PATH = PROJECT_ROOT / "app"
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app.layout import (  # noqa: E402
	create_animal_missing_page,
	create_animal_page,
	create_home_page,
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
RANCH_ANIMAL_HINTS = {
	"kudu": "greater-kudu",
	"springbok": "springbok",
	"zebra": "plains-zebra",
	"giraffe": "cape-giraffe",
	"wildebeest": "blue-wildebeest",
	"gnu": "blue-wildebeest",
	"ostrich": "common-ostrich",
}
OUTSIDE_RANCH_HINTS = (
	"lion",
	"leopard",
	"cheetah",
	"rhino",
	"hippo",
	"crocodile",
	"elephant",
	"penguin",
	"whale",
	"shark",
)


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
	action_href: str | None = None
	action_label: str | None = None


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


def select_ranch_animal(file_name: str, image_bytes: bytes) -> RanchAnimal:
	lowered_name = file_name.lower()
	for keyword, animal_id in RANCH_ANIMAL_HINTS.items():
		if keyword in lowered_name:
			return RANCH_ANIMALS_BY_ID[animal_id]

	index = int(hashlib.sha256(image_bytes).hexdigest()[8:10], 16) % len(RANCH_ANIMALS)
	return RANCH_ANIMALS[index]


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
		reasons.append(f"Matched ranch animal route: {animal.name}.")
		return PredictionResult(
			title=animal.name,
			badge_text="Exists on the ranch",
			confidence=confidence,
			badge_class="result-badge--positive",
			fill_class="confidence-fill--positive",
			summary=(
				"This upload matches one of the South African animals currently listed on Bosvelder Ranch. Open the animal page to view the profile and ranch details."
			),
			reasons=tuple(reasons),
			mode_label=mode_label,
			details=(
				("Species", animal.scientific_name),
				("Habitat zone", animal.habitat_zone),
				("Best viewing", animal.best_viewing),
			),
			note="This is still a demo result. When the real model is connected, the page flow can stay the same while only the recognition step changes.",
			action_href=animal.page_href,
			action_label="Open animal page",
		)

	return PredictionResult(
		title="Animal not found on this ranch",
		badge_text="Not on the ranch",
		confidence=confidence,
		badge_class="result-badge--negative",
		fill_class="confidence-fill--negative",
		summary=(
			"We could not confidently match this upload to the current Bosvelder Ranch animal list. Try another image or upload a clearer photo of the species."
		),
		reasons=tuple(reasons),
		mode_label=mode_label,
		details=(
			("Catalog scope", "Bosvelder Ranch species list"),
			("Status", "No confident match"),
			("Next step", "Try another image"),
		),
		note="This negative result is also placeholder logic until the trained classifier is wired in.",
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
		create_home_page(RANCH_ANIMALS),
		create_animal_page(RANCH_ANIMALS[0]),
		create_animal_missing_page(),
	]
)


@app.callback(
	Output("page-content", "children"),
	Input("url", "pathname"),
	Input("url", "search"),
)
def render_page(pathname: str | None, search: str | None):
	if pathname == "/animals":
		query_params = parse_qs((search or "").lstrip("?"))
		animal = get_ranch_animal(query_params.get("animal", [None])[0])
		if animal is None:
			return create_animal_missing_page()

		return create_animal_page(animal)

	return create_home_page(RANCH_ANIMALS)


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
		result_card(prediction),
	)


if __name__ == "__main__":
	app.run(
		host="0.0.0.0",
		port=int(os.getenv("PORT", "8050")),
		debug=os.getenv("DASH_DEBUG", "false").lower() == "true",
	)
