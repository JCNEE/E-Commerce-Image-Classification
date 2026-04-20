from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnimalLocation:
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


# Add one or more AnimalLocation entries per animal_id once your ranch coordinates are ready.
ANIMAL_LOCATION_DATA: dict[str, tuple[AnimalLocation, ...]] = {
	"greater-kudu": (),
	"springbok": (),
	"plains-zebra": (),
	"cape-giraffe": (),
	"blue-wildebeest": (),
	"common-ostrich": (),
}


def get_animal_locations(animal_id: str | None) -> tuple[AnimalLocation, ...]:
	if not animal_id:
		return ()
	return ANIMAL_LOCATION_DATA.get(animal_id, ())


def get_animal_location(animal_id: str | None, location_id: str | None) -> AnimalLocation | None:
	if not animal_id or not location_id:
		return None

	for location in get_animal_locations(animal_id):
		if location.location_id == location_id:
			return location

	return None