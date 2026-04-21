from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode


ASSETS_PATH = Path(__file__).resolve().parent / "assets"


def _build_placeholder_image(title: str, accent_start: str, accent_end: str) -> str:
	svg = f"""
	<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 700" role="img" aria-label="{title}">
	  <defs>
	    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
	      <stop offset="0%" stop-color="{accent_start}" />
	      <stop offset="100%" stop-color="{accent_end}" />
	    </linearGradient>
	  </defs>
	  <rect width="1200" height="700" fill="url(#bg)" />
	  <circle cx="1010" cy="150" r="190" fill="rgba(255,255,255,0.12)" />
	  <circle cx="170" cy="560" r="220" fill="rgba(255,255,255,0.10)" />
	  <rect x="84" y="438" width="1032" height="166" rx="34" fill="rgba(34,22,15,0.28)" />
	  <text x="110" y="512" fill="#fff9f1" font-size="84" font-family="Georgia, serif" font-weight="700">{title}</text>
	  <text x="112" y="570" fill="#f7e8d4" font-size="28" font-family="Arial, sans-serif">Temporary catalog artwork placeholder</text>
	</svg>
	""".strip()
	return "data:image/svg+xml;charset=utf-8," + quote(svg)


def _asset_or_placeholder(filename: str, title: str, accent_start: str, accent_end: str) -> str:
	asset_path = ASSETS_PATH / filename
	if asset_path.exists():
		return f"/assets/assets/{filename}"
	return _build_placeholder_image(title, accent_start, accent_end)


def normalise_animal_text(value: str | None) -> str:
	if not value:
		return ""

	text = value.strip().lower().replace("_", " ").replace("-", " ")
	text = text.replace("'", "")
	text = re.sub(r"[^a-z0-9]+", " ", text)
	return " ".join(text.split())


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
		"Greater kudu are one of the signature sale species in the catalog. Bulls carry long spiral horns while cows and young animals move in looser groups through thicker bush, usually browsing on leaves and pods rather than open grass.",
		"Acacia ridge and mixed bushveld line",
		"Early morning near shaded browsing paths",
		_asset_or_placeholder("greater-kudu.jpg", "Greater Kudu", "#7f5d3d", "#cf9b5a"),
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
		"Springbok are among the most recognisable South African antelope. They are usually seen out in the open, grazing in loose groups and suddenly springing into the air when startled.",
		"Open grass flats and short-grazing camps",
		"Late afternoon across the sunlit plains",
		_asset_or_placeholder("springbok.jpg", "Springbok", "#8c5e34", "#d8b06a"),
		(
			"Watch for the chestnut stripe along each side of the body.",
			"Pronking jumps are a common sign that the herd is alert.",
			"They usually prefer the more open grazing areas of the ranch.",
		),
	),
	RanchAnimal(
		"cape-giraffe",
		"Cape Giraffe",
		"Giraffa giraffa giraffa",
		"Browsers",
		"A towering browser that strips leaves from the upper acacia canopy.",
		"Cape giraffe are among the easiest animals to spot in the catalog because they rise above the thorn trees. Their pale patches and long silhouette make them one of the most recognisable listed species.",
		"Tall acacia belt on the northern loop",
		"Early afternoon where tree canopies stay green",
		_asset_or_placeholder("cape-giraffe.jpg", "Cape Giraffe", "#a56f3a", "#e1bc73"),
		(
			"Notice the patch pattern and the dark tail tuft moving while they feed.",
			"They often browse at heights other animals cannot reach.",
			"Groups may include calves staying close to the adults on the edges of the trees.",
		),
	),
	RanchAnimal(
		"buffalo",
		"Buffalo",
		"Syncerus caffer",
		"Heavy grazers",
		"A broad-fronted grazer with a powerful boss and a slow, deliberate herd movement.",
		"Buffalo add weight and presence to the sale catalog. They prefer thicker grasslands and water access, moving in compact groups that stay tightly aware of movement around them.",
		"Waterline grass camps and dense grazing blocks",
		"Morning around water access routes",
		_asset_or_placeholder("buffalo.jpg", "Buffalo", "#5c4636", "#b38155"),
		(
			"The heavy horn boss gives mature animals a blocky forehead profile.",
			"They tend to move as a tighter unit than most antelope species.",
			"Dark body colour and stocky shoulders make them stand out in mixed herds.",
		),
	),
	RanchAnimal(
		"rhinoceros",
		"Rhino",
		"Ceratotherium simum / Diceros bicornis",
		"Big five",
		"A massive grazer with a deep barrel body, thick skin, and a clear facial horn.",
		"Rhino are treated as a premium listed species in the catalog. They are usually identified by their heavy shape, lowered head posture, and the very distinct horn profile visible even at distance.",
		"Open browse edge and protected grass sections",
		"Cool daylight hours near open clearings",
		_asset_or_placeholder("rhino.jpg", "Rhino", "#67625d", "#b9a694"),
		(
			"The horn profile is the fastest visual cue, even in side silhouette.",
			"They hold a much heavier body line than buffalo or wildebeest.",
			"A lowered head and deliberate pace are common field markers.",
		),
	),
	RanchAnimal(
		"plains-zebra",
		"Plains Zebra",
		"Equus quagga",
		"Grazers",
		"A bold black-and-cream grazer that moves in family groups across the plains.",
		"Plains zebra bring strong movement and contrast to the catalog. Their stripe pattern is unique to each animal, but the herd behaviour is just as helpful when identifying them.",
		"Central grazing paddocks and waterline paths",
		"Mid-morning around open water points",
		_asset_or_placeholder("plains-zebra.jpg", "Plains Zebra", "#6d5d52", "#ddd3c8"),
		(
			"Each zebra has its own stripe pattern even though the herd appears uniform from a distance.",
			"They often stand in loose lines while grazing or resting.",
			"Their rounded ears and upright mane help distinguish them from antelope at range.",
		),
	),
	RanchAnimal(
		"common-ostrich",
		"Common Ostrich",
		"Struthio camelus",
		"Birdlife",
		"The world's largest bird, built for speed across dry South African ground.",
		"Ostriches bring a very different silhouette to the sale catalog. Their height, bare legs, and long neck make them easy to identify even from a distance.",
		"Dry grass camps and open fence lines",
		"Warm daylight hours on open ground",
		_asset_or_placeholder("common-ostrich.jpg", "Common Ostrich", "#8b6640", "#ceb087"),
		(
			"The long neck and powerful legs are visible even from far away.",
			"They tend to prefer open areas with a clear line of sight.",
			"Fast running is one of the clearest behavioural cues when they are disturbed.",
		),
	),
	RanchAnimal(
		"elephant",
		"Elephant",
		"Loxodonta africana",
		"Big five",
		"A large grey browser with a swinging trunk and a broad, unmistakable silhouette.",
		"Elephants are one of the clearest large-animal listings in the catalog. Their trunk, ear shape, and very high body mass make them straightforward to separate from every other sold species.",
		"Riverine browse and dense shade corridors",
		"Morning or late afternoon near water and trees",
		_asset_or_placeholder("elephant.jpg", "Elephant", "#72675b", "#c1ad93"),
		(
			"The trunk remains the strongest recognition cue from almost any angle.",
			"Broad ears and a rounded forehead separate them from rhino at distance.",
			"They often move through tree lines while feeding continuously.",
		),
	),
	RanchAnimal(
		"lion",
		"Lion",
		"Panthera leo",
		"Big cats",
		"A powerful apex predator with a deep chest, broad head, and unmistakable cat silhouette.",
		"Lions replace blue wildebeest in the sale catalog and bring a very different visual profile to the site. Their muscular frame, forward-facing gaze, and, in males, a full mane make them one of the easiest listed species to separate from the hoofed animals on the ranch pages.",
		"Protected predator camps and open bushveld edges",
		"Cool mornings and late afternoons near shade lines",
		_asset_or_placeholder("lion.jpg", "Lion", "#8a6138", "#d0a267"),
		(
			"The broad muzzle and cat-like posture stand apart from every grazing species in the catalog.",
			"Adult males show the clearest mane profile around the head, neck, and chest.",
			"They are usually identified by slow, deliberate movement followed by short bursts of attention or patrol.",
		),
	),
	RanchAnimal(
		"hippopotamus",
		"Hippopotamus",
		"Hippopotamus amphibius",
		"Waterline",
		"A large semi-aquatic grazer with a heavy jaw and a broad-backed river silhouette.",
		"Hippopotamus are handled as a waterline listing in the catalog. They are recognised by their barrel shape, low profile in water, and a very broad head with a square muzzle.",
		"Dam edges, river bends, and still water access points",
		"Late afternoon near calm water",
		_asset_or_placeholder("hippo.jpg", "Hippopotamus", "#60524b", "#a4897f"),
		(
			"The broad square muzzle is more diagnostic than body colour.",
			"They often show only the head and back line above water.",
			"A low-set body and very short legs separate them from elephant or rhino.",
		),
	),
)

RANCH_ANIMALS_BY_ID = {animal.animal_id: animal for animal in RANCH_ANIMALS}

_ANIMAL_ALIASES = {
	"african lion": "lion",
	"buffalo": "buffalo",
	"buffaloes": "buffalo",
	"buffalos": "buffalo",
	"cape giraffe": "cape-giraffe",
	"common ostrich": "common-ostrich",
	"elephant": "elephant",
	"elephants": "elephant",
	"giraffe": "cape-giraffe",
	"giraffes": "cape-giraffe",
	"greater kudu": "greater-kudu",
	"hippo": "hippopotamus",
	"hippopotami": "hippopotamus",
	"hippopotamus": "hippopotamus",
	"hippopotomaus": "hippopotamus",
	"kudu": "greater-kudu",
	"kudus": "greater-kudu",
	"lion": "lion",
	"lions": "lion",
	"ostrich": "common-ostrich",
	"ostriches": "common-ostrich",
	"plains zebra": "plains-zebra",
	"rhino": "rhinoceros",
	"rhinos": "rhinoceros",
	"rhinoceros": "rhinoceros",
	"rhinoceroses": "rhinoceros",
	"springbok": "springbok",
	"springbokke": "springbok",
	"springboks": "springbok",
	"zebra": "plains-zebra",
	"zebras": "plains-zebra",
}
ANIMAL_ALIASES = {
	normalise_animal_text(alias): animal_id for alias, animal_id in _ANIMAL_ALIASES.items()
}
ANIMAL_ALIAS_PHRASES = tuple(sorted(ANIMAL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True))


def get_ranch_animal(animal_id: str | None) -> RanchAnimal | None:
	if not animal_id:
		return None
	return RANCH_ANIMALS_BY_ID.get(animal_id)


def infer_catalog_animal_id(text: str | None) -> str | None:
	normalised = normalise_animal_text(text)
	if not normalised:
		return None

	haystack = f" {normalised} "
	for alias, animal_id in ANIMAL_ALIAS_PHRASES:
		if f" {alias} " in haystack:
			return animal_id

	return ANIMAL_ALIASES.get(normalised)


def infer_catalog_animal(text: str | None) -> RanchAnimal | None:
	return get_ranch_animal(infer_catalog_animal_id(text))