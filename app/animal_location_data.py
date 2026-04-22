"""Saved ranch-location records and helpers for animal and district pages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnimalLocation:
	"""Static location metadata for a hunt site shown in the catalog experience."""

	location_id: str
	area_name: str
	ranch_name: str
	latitude: float
	longitude: float
	area_summary: str
	district_story: str
	highlights: tuple[str, ...] = ()
	district_name: str | None = None
	province_name: str | None = None


def _build_area_summary(area_name: str, species_name: str, district_name: str, province_name: str) -> str:
	"""Create the short location summary stored with each ranch area."""

	return (
		f"{area_name} is stored in the app as a {species_name.lower()} hunting location "
		f"in {district_name}, {province_name}."
	)


def _build_district_story(area_name: str, species_name: str, district_name: str, province_name: str) -> str:
	"""Create the lead paragraph used on the district detail page."""

	return (
		f"This district view centers on {district_name} in {province_name}, where {area_name} is "
		f"listed as one of the configured {species_name.lower()} hunting locations."
	)


def _build_highlights(species_name: str, district_name: str, province_name: str) -> tuple[str, ...]:
	"""Generate reusable bullet-point highlights for a district page."""

	return (
		f"Configured species: {species_name}.",
		f"District focus: {district_name}, {province_name}.",
		"Use the animal page cards to open district detail without relying on interactive map clicks.",
	)


def _location(
	location_id: str,
	area_name: str,
	ranch_name: str,
	latitude: float,
	longitude: float,
	district_name: str,
	province_name: str,
	species_name: str,
) -> AnimalLocation:
	"""Build a location record while keeping summary text generation in one place."""

	return AnimalLocation(
		location_id=location_id,
		area_name=area_name,
		ranch_name=ranch_name,
		latitude=latitude,
		longitude=longitude,
		area_summary=_build_area_summary(area_name, species_name, district_name, province_name),
		district_story=_build_district_story(area_name, species_name, district_name, province_name),
		highlights=_build_highlights(species_name, district_name, province_name),
		district_name=district_name,
		province_name=province_name,
	)


# Static hunt-location data grouped by the sold-animal identifiers used in routing.
ANIMAL_LOCATION_DATA: dict[str, tuple[AnimalLocation, ...]] = {
	"buffalo": (
		_location(
			"eulalie-hunting-safari-lodge",
			"Eulalie Hunting Safari Lodge",
			"Eulalie Hunting Safari Lodge",
			-23.409938444942657,
			28.26452869846609,
			"Waterberg District",
			"Limpopo",
			"Buffalo",
		),
		_location(
			"buffalo-kloof-conservancy",
			"Buffalo Kloof Conservancy",
			"Rhino Conservation | Buffalo Kloof Conservancy South Africa",
			-33.39031017517954,
			26.571216107096653,
			"Sarah Baartman District",
			"Eastern Cape",
			"Buffalo",
		),
	),
	"elephant": (
		_location(
			"huntshoek-safaris",
			"Huntshoek Safaris",
			"Available Species - Huntshoek Safaris",
			-33.251413239731,
			26.966501833066584,
			"Sarah Baartman District",
			"Eastern Cape",
			"Elephant",
		),
		_location(
			"rdb-safaris",
			"RDB Safaris",
			"All Packages - RDB Safaris",
			-24.288059541152307,
			28.675731477827842,
			"Waterberg District",
			"Limpopo",
			"Elephant",
		),
	),
	"greater-kudu": (
		_location(
			"kudu-safari-africa",
			"Kudu Safari Africa",
			"Home - Kudu Safari Africa",
			-28.88138378184504,
			24.60424751762567,
			"Frances Baard District",
			"Northern Cape",
			"Greater Kudu",
		),
		_location(
			"kudu-valley-safari-lodge",
			"Kudu Valley Safari Lodge",
			"Kudu Valley Safari Lodge - Africa's Premier Safari Destination",
			-24.501778605478364,
			27.41405483197084,
			"Waterberg District",
			"Limpopo",
			"Greater Kudu",
		),
	),
	"hippopotamus": (
		_location(
			"rdb-safaris",
			"RDB Safaris",
			"All Packages - RDB Safaris",
			-24.288059541152307,
			28.675731477827842,
			"Waterberg District",
			"Limpopo",
			"Hippopotamus",
		),
		_location(
			"huntshoek-safaris",
			"Huntshoek Safaris",
			"Available Species - Huntshoek Safaris",
			-33.251413239731,
			26.966501833066584,
			"Sarah Baartman District",
			"Eastern Cape",
			"Hippopotamus",
		),
	),
	"springbok": (
		_location(
			"riverstone-game-farm",
			"Riverstone Game Farm, Ladismith",
			"105 Ockerskraal Plaas, Ladismith, 6655",
			-33.70102592144614,
			21.169519186504637,
			"Eden District",
			"Western Cape",
			"Springbok",
		),
		_location(
			"frontier-safaris",
			"Frontier Safaris",
			"South Africa - Frontier Safaris",
			-33.2753870432294,
			26.15210484014442,
			"Sarah Baartman District",
			"Eastern Cape",
			"Springbok",
		),
	),
	"rhinoceros": (
		_location(
			"dinokeng-game-reserve",
			"Dinokeng Game Reserve",
			"Dinokeng Game Reserve Conservation Efforts",
			-25.400949894275907,
			28.307066726516453,
			"City of Tshwane Metropolitan",
			"Gauteng",
			"Rhinoceros",
		),
		_location(
			"buffalo-kloof-conservancy",
			"Buffalo Kloof Conservancy",
			"Rhino Conservation | Buffalo Kloof Conservancy South Africa",
			-33.39031017517954,
			26.571216107096653,
			"Sarah Baartman District",
			"Eastern Cape",
			"Rhinoceros",
		),
	),
	"plains-zebra": (
		_location(
			"iconic-african-trophy",
			"Iconic African Trophy",
			"Zebra Hunting South Africa | Iconic African Trophy",
			-23.418724706666477,
			28.256120252637054,
			"Waterberg District",
			"Limpopo",
			"Plains Zebra",
		),
		_location(
			"africa-hunt-lodge",
			"Africa Hunt Lodge",
			"Hunting Packages | Africa Hunt Lodge | South Africa Hunts",
			-24.297060265268865,
			27.43853325598847,
			"Waterberg District",
			"Limpopo",
			"Plains Zebra",
		),
	),
	"cape-giraffe": (
		_location(
			"african-hunt-lodge",
			"African Hunt Lodge",
			"Farm Witklip, R510, Thabazimbi, 0380",
			-24.29414196250198,
			27.43879286857414,
			"Waterberg District",
			"Limpopo",
			"Cape Giraffe",
		),
		_location(
			"gaspare-spanio-safaris",
			"Gaspare Spanio Safaris",
			"Farm Kudusrand, Marken, 0605",
			-23.418552006205665,
			28.256077628834884,
			"Waterberg District",
			"Limpopo",
			"Cape Giraffe",
		),
	),
	"common-ostrich": (
		_location(
			"gaspare-spanio-safaris",
			"Gaspare Spanio Safaris",
			"Farm Kudusrand, Marken, 0605",
			-23.418552006205665,
			28.256077628834884,
			"Waterberg District",
			"Limpopo",
			"Common Ostrich",
		),
		_location(
			"moreson-ranch",
			"Moreson Ranch",
			"310, Vrede, 9835",
			-27.515841877271036,
			29.10872327116511,
			"Thabo Mofutsanyana District",
			"Free State",
			"Common Ostrich",
		),
	),
	"lion": (
		_location(
			"lion-and-safari-park",
			"Lion & Safari Park",
			"R512 Pelindaba Rd, Broederstroom, 0240",
			-25.83214839638195,
			27.88876224233024,
			"Bojanala Platinum District",
			"North West",
			"Lion",
		),
		_location(
			"bothongo-rhino-and-lion-nature-reserve",
			"Bothongo Rhino & Lion Nature Reserve",
			"520 Kromdraai Road, Kromdraai, Krugersdorp, 1739",
			-25.971446842415066,
			27.79197188650463,
			"West Rand District",
			"Gauteng",
			"Lion",
		),
	),
}


def get_animal_locations(animal_id: str | None) -> tuple[AnimalLocation, ...]:
	"""Return every saved hunt location for a routed animal page."""

	if not animal_id:
		return ()
	return ANIMAL_LOCATION_DATA.get(animal_id, ())


def get_animal_location(animal_id: str | None, location_id: str | None) -> AnimalLocation | None:
	"""Return a single location record for the selected district page."""

	if not animal_id or not location_id:
		return None

	for location in get_animal_locations(animal_id):
		if location.location_id == location_id:
			return location

	return None