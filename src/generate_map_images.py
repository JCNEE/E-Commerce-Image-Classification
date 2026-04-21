from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("USE_GEOJSON_MAPS", "true")

from app.animal_catalog import RANCH_ANIMALS  # noqa: E402
from app.animal_location_data import get_animal_locations  # noqa: E402
from app.animal_range_map import (  # noqa: E402
	ANIMAL_RANGE,
	GENERATED_MAPS_PATH,
	build_animal_location_map,
	build_district_detail_map,
	build_sa_range_map,
	get_animal_location_map_asset_path,
	get_district_map_asset_path,
	get_range_map_asset_path,
)


def ensure_parent_dir(file_path: Path) -> None:
	file_path.parent.mkdir(parents=True, exist_ok=True)


def write_map_image(
	figure,
	output_path: Path,
	label: str,
	primary_scale: int = 2,
	fallback_scale: int = 1,
) -> None:
	if figure is None:
		print(f"Skipping {label}: no figure was built.")
		return

	ensure_parent_dir(output_path)
	try:
		figure.write_image(str(output_path), format="png", scale=primary_scale)
	except ValueError as exc:
		raise RuntimeError(
			"Plotly image export requires kaleido. Install it with 'pip install kaleido' or add it to your tooling environment."
		) from exc
	except TimeoutError:
		if fallback_scale == primary_scale:
			print(f"Skipping {label}: export timed out at scale {primary_scale}.")
			return

		print(
			f"Retrying {label}: export timed out at scale {primary_scale}, retrying at scale {fallback_scale}."
		)
		try:
			figure.write_image(str(output_path), format="png", scale=fallback_scale)
		except TimeoutError:
			print(f"Skipping {label}: export timed out again at scale {fallback_scale}.")
			return

	print(f"Wrote {label}: {output_path.relative_to(PROJECT_ROOT)}")


def generate_range_maps() -> None:
	for species_name in sorted(ANIMAL_RANGE):
		output_path = get_range_map_asset_path(species_name)
		if output_path is None:
			continue
		write_map_image(build_sa_range_map(species_name), output_path, f"range map for {species_name}")


def generate_animal_and_district_maps() -> None:
	for animal in RANCH_ANIMALS:
		locations = get_animal_locations(animal.animal_id)
		animal_map = build_animal_location_map(animal.name, locations)
		if animal_map is None:
			animal_map = build_sa_range_map(animal.name)

		animal_output_path = get_animal_location_map_asset_path(animal.animal_id)
		if animal_output_path is not None:
			write_map_image(animal_map, animal_output_path, f"animal map for {animal.animal_id}")

		for location in locations:
			district_output_path = get_district_map_asset_path(animal.animal_id, location.location_id)
			if district_output_path is None:
				continue
			write_map_image(
				build_district_detail_map(location),
				district_output_path,
				f"district map for {animal.animal_id}/{location.location_id}",
				primary_scale=1,
				fallback_scale=1,
			)


def main() -> None:
	GENERATED_MAPS_PATH.mkdir(parents=True, exist_ok=True)
	generate_range_maps()
	generate_animal_and_district_maps()


if __name__ == "__main__":
	main()