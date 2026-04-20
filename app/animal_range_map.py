from __future__ import annotations

import json
from dataclasses import dataclass

from functools import lru_cache

from pathlib import Path

import plotly.graph_objects as go

from app.animal_location_data import AnimalLocation


ASSETS_PATH = Path(__file__).resolve().parent / "assets"
GEOJSON_PATH = ASSETS_PATH / "south-africa.geojson"

SA_PROVINCES = (
	"Western Cape",
	"Eastern Cape",
	"Northern Cape",
	"Free State",
	"KwaZulu-Natal",
	"North West",
	"Gauteng",
	"Mpumalanga",
	"Limpopo",
)

PROVINCE_CENTROIDS = {
	"Western Cape": (-33.9249, 18.4241),
	"Eastern Cape": (-32.2968, 26.4194),
	"Northern Cape": (-29.0467, 21.8569),
	"Free State": (-28.4541, 26.7968),
	"KwaZulu-Natal": (-28.5305, 30.8958),
	"North West": (-26.6639, 25.2838),
	"Gauteng": (-26.2708, 28.1123),
	"Mpumalanga": (-25.5653, 30.5279),
	"Limpopo": (-23.4013, 29.4179),
}

# The supplied GeoJSON contains district and metropolitan boundaries rather than
# province polygons, so province highlighting is expanded to all matching child features.
PROVINCE_DISTRICTS = {
	"Western Cape": (
		"City of Cape Town",
		"West Coast District",
		"Cape Winelands District",
		"Overberg District",
		"Eden District",
		"Central Karoo District",
	),
	"Eastern Cape": (
		"Buffalo City Metropolitan",
		"Nelson Mandela Bay Metropolitan",
		"Alfred Nzo District",
		"Amathole District",
		"Chris Hani District",
		"Joe Gqabi District",
		"O.R. Tambo District",
		"Sarah Baartman District",
	),
	"Northern Cape": (
		"Frances Baard District",
		"John Taolo Gaetsewe District",
		"Namakwa District",
		"Pixley ka Seme District",
		"ZF Mgcawu District",
	),
	"Free State": (
		"Mangaung Metropolitan",
		"Fezile Dabi District",
		"Lejweleputswa District",
		"Thabo Mofutsanyana District",
		"Xhariep District",
	),
	"KwaZulu-Natal": (
		"eThekwini Metropolitan",
		"Amajuba District",
		"iLembe District",
		"Sisonke District",
		"Ugu District",
		"uMgungundlovu District",
		"Umkhanyakude District",
		"Umzinyathi District",
		"Uthukela District",
		"uThungulu District",
		"Zululand District",
	),
	"North West": (
		"Bojanala Platinum District",
		"Dr Kenneth Kaunda District",
		"Dr Ruth Segomotsi Mompati District",
		"Ngaka Modiri Molema District",
	),
	"Gauteng": (
		"City of Johannesburg Metropolitan",
		"City of Tshwane Metropolitan",
		"Ekurhuleni Metropolitan",
		"Sedibeng District",
		"West Rand District",
	),
	"Mpumalanga": (
		"Ehlanzeni District",
		"Gert Sibande District",
		"Nkangala District",
	),
	"Limpopo": (
		"Capricorn District",
		"Mopani District",
		"Sekhukhune District",
		"Vhembe District",
		"Waterberg District",
	),
}
DISTRICT_TO_PROVINCE = {
	district_name: province_name
	for province_name, district_names in PROVINCE_DISTRICTS.items()
	for district_name in district_names
}

SPECIES_DISPLAY_NAMES = {
	"buffalo": "Buffalo",
	"cheetah": "Cheetah",
	"elephant": "Elephant",
	"giraffe": "Giraffe",
	"hippopotamus": "Hippopotamus",
	"kudu": "Kudu",
	"leopard": "Leopard",
	"lion": "Lion",
	"ostrich": "Ostrich",
	"rhinoceros": "Rhinoceros",
	"springbok": "Springbok",
	"warthog": "Warthog",
	"wildebeest": "Wildebeest",
	"zebra": "Zebra",
}

SPECIES_ALIASES = {
	"blue wildebeest": "wildebeest",
	"cape giraffe": "giraffe",
	"common ostrich": "ostrich",
	"gnu": "wildebeest",
	"greater kudu": "kudu",
	"hippo": "hippopotamus",
	"plains zebra": "zebra",
	"rhino": "rhinoceros",
}

ANIMAL_RANGE = {
	"lion": ("Limpopo", "Mpumalanga", "KwaZulu-Natal", "North West"),
	"elephant": ("Limpopo", "Mpumalanga", "KwaZulu-Natal", "Eastern Cape"),
	"rhinoceros": ("Limpopo", "Mpumalanga", "KwaZulu-Natal"),
	"zebra": ("Limpopo", "Mpumalanga", "Northern Cape", "Eastern Cape"),
	"giraffe": ("Limpopo", "Mpumalanga", "KwaZulu-Natal", "North West"),
	"springbok": ("Northern Cape", "Western Cape", "Free State", "Eastern Cape"),
	"cheetah": ("Limpopo", "North West", "Northern Cape", "Mpumalanga"),
	"buffalo": ("Limpopo", "Mpumalanga", "KwaZulu-Natal"),
	"warthog": ("Limpopo", "Mpumalanga", "KwaZulu-Natal", "North West"),
	"kudu": ("Limpopo", "Mpumalanga", "Eastern Cape", "Northern Cape"),
	"leopard": ("Limpopo", "Mpumalanga", "Western Cape", "KwaZulu-Natal"),
	"hippopotamus": ("Limpopo", "Mpumalanga", "KwaZulu-Natal"),
	"wildebeest": ("Limpopo", "Mpumalanga", "North West", "Free State"),
	"ostrich": ("Western Cape", "Northern Cape", "Eastern Cape", "Free State"),
}


@dataclass(frozen=True)
class SpeciesRangeContext:
	canonical_name: str
	display_name: str
	provinces: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedAnimalLocation:
	location: AnimalLocation
	district_name: str | None
	province_name: str | None


@lru_cache(maxsize=1)
def load_sa_geojson() -> dict | None:
	if not GEOJSON_PATH.exists():
		return None

	try:
		with GEOJSON_PATH.open("r", encoding="utf-8") as geojson_file:
			return json.load(geojson_file)
	except (OSError, json.JSONDecodeError):
		return None


@lru_cache(maxsize=1)
def get_geojson_features_by_name() -> dict[str, dict]:
	geojson = load_sa_geojson()
	if geojson is None:
		return {}

	return {
		feature.get("properties", {}).get("name"): feature
		for feature in geojson.get("features", [])
		if feature.get("properties", {}).get("name")
	}


def iter_feature_polygons(feature: dict) -> tuple[list, ...]:
	geometry = feature.get("geometry") or {}
	geometry_type = geometry.get("type")
	coordinates = geometry.get("coordinates") or []
	if geometry_type == "Polygon":
		return (coordinates,)
	if geometry_type == "MultiPolygon":
		return tuple(coordinates)
	return ()


def point_in_ring(longitude: float, latitude: float, ring: list[list[float]]) -> bool:
	inside = False
	if len(ring) < 3:
		return False

	for index, point in enumerate(ring):
		next_point = ring[(index + 1) % len(ring)]
		lon_a, lat_a = point[0], point[1]
		lon_b, lat_b = next_point[0], next_point[1]
		intersects = ((lat_a > latitude) != (lat_b > latitude)) and (
			longitude < (lon_b - lon_a) * (latitude - lat_a) / ((lat_b - lat_a) or 1e-12) + lon_a
		)
		if intersects:
			inside = not inside

	return inside


def point_in_polygon(longitude: float, latitude: float, polygon: list[list[list[float]]]) -> bool:
	if not polygon:
		return False

	outer_ring = polygon[0]
	if not point_in_ring(longitude, latitude, outer_ring):
		return False

	for inner_ring in polygon[1:]:
		if point_in_ring(longitude, latitude, inner_ring):
			return False

	return True


def find_district_for_coordinate(latitude: float, longitude: float, geojson: dict | None = None) -> str | None:
	geojson = geojson or load_sa_geojson()
	if geojson is None:
		return None

	for feature in geojson.get("features", []):
		feature_name = feature.get("properties", {}).get("name")
		if not feature_name:
			continue

		for polygon in iter_feature_polygons(feature):
			if point_in_polygon(longitude, latitude, polygon):
				return feature_name

	return None


def resolve_animal_location(
	location: AnimalLocation,
	geojson: dict | None = None,
) -> ResolvedAnimalLocation:
	geojson = geojson or load_sa_geojson()
	district_name = location.district_name or find_district_for_coordinate(
		location.latitude,
		location.longitude,
		geojson,
	)
	province_name = location.province_name or DISTRICT_TO_PROVINCE.get(district_name or "")
	return ResolvedAnimalLocation(
		location=location,
		district_name=district_name,
		province_name=province_name,
	)


def resolve_animal_locations(
	locations: tuple[AnimalLocation, ...],
	geojson: dict | None = None,
) -> tuple[ResolvedAnimalLocation, ...]:
	return tuple(resolve_animal_location(location, geojson) for location in locations)


def iter_coordinate_pairs(values) -> tuple[tuple[float, float], ...]:
	if not isinstance(values, list) or not values:
		return ()

	first_item = values[0]
	if isinstance(first_item, (int, float)):
		if len(values) >= 2:
			return ((float(values[0]), float(values[1])),)
		return ()

	coordinate_pairs: list[tuple[float, float]] = []
	for value in values:
		coordinate_pairs.extend(iter_coordinate_pairs(value))
	return tuple(coordinate_pairs)


def get_feature_bounds(feature: dict) -> tuple[float, float, float, float] | None:
	coordinate_pairs = iter_coordinate_pairs(feature.get("geometry", {}).get("coordinates") or [])
	if not coordinate_pairs:
		return None

	longitudes = [pair[0] for pair in coordinate_pairs]
	latitudes = [pair[1] for pair in coordinate_pairs]
	return (min(longitudes), max(longitudes), min(latitudes), max(latitudes))


def normalise_species_name(species_name: str | None) -> str | None:
	if not species_name:
		return None

	key = species_name.strip().lower().replace("_", " ").replace("-", " ")
	key = " ".join(key.split())
	canonical_name = SPECIES_ALIASES.get(key, key)
	if canonical_name not in ANIMAL_RANGE:
		return None

	return canonical_name


def get_species_range_context(species_name: str | None) -> SpeciesRangeContext | None:
	canonical_name = normalise_species_name(species_name)
	if canonical_name is None:
		return None

	return SpeciesRangeContext(
		canonical_name=canonical_name,
		display_name=SPECIES_DISPLAY_NAMES.get(canonical_name, canonical_name.title()),
		provinces=ANIMAL_RANGE[canonical_name],
	)


def build_sa_geojson_range_map(range_context: SpeciesRangeContext, geojson: dict) -> go.Figure:
	highlighted_provinces = set(range_context.provinces)
	locations: list[str] = []
	zone_values: list[int] = []
	hover_text: list[str] = []

	for feature in geojson.get("features", []):
		feature_name = feature.get("properties", {}).get("name")
		if not feature_name:
			continue

		province_name = DISTRICT_TO_PROVINCE.get(feature_name)
		is_highlighted = province_name in highlighted_provinces if province_name else False
		status_text = "Highlighted for" if is_highlighted else "Not highlighted for"

		locations.append(feature_name)
		zone_values.append(1 if is_highlighted else 0)
		hover_text.append(
			f"<b>{feature_name}</b><br>Province: {province_name or 'Unknown'}<br>{status_text} {range_context.display_name}"
		)

	fig = go.Figure(
		go.Choropleth(
			geojson=geojson,
			locations=locations,
			z=zone_values,
			featureidkey="properties.name",
			text=hover_text,
			hovertemplate="%{text}<extra></extra>",
			colorscale=[
				[0.0, "#d8c7ae"],
				[0.4999, "#d8c7ae"],
				[0.5, "#2f6d4d"],
				[1.0, "#2f6d4d"],
			],
			showscale=False,
			zmin=0,
			zmax=1,
			marker_line_color="#f8f0e1",
			marker_line_width=0.8,
		)
	)
	fig.update_geos(
		fitbounds="locations",
		visible=False,
		projection_type="mercator",
		bgcolor="rgba(0, 0, 0, 0)",
	)
	fig.update_layout(
		height=360,
		margin={"r": 0, "t": 0, "l": 0, "b": 0},
		paper_bgcolor="rgba(0, 0, 0, 0)",
		plot_bgcolor="rgba(0, 0, 0, 0)",
	)

	return fig


def build_neutral_geojson_map(geojson: dict) -> go.Figure:
	locations: list[str] = []
	hover_text: list[str] = []

	for feature in geojson.get("features", []):
		feature_name = feature.get("properties", {}).get("name")
		if not feature_name:
			continue

		locations.append(feature_name)
		hover_text.append(f"<b>{feature_name}</b><extra></extra>")

	fig = go.Figure(
		go.Choropleth(
			geojson=geojson,
			locations=locations,
			z=[0 for _ in locations],
			featureidkey="properties.name",
			text=hover_text,
			hovertemplate="%{text}",
			colorscale=[[0.0, "#efe4d2"], [1.0, "#efe4d2"]],
			showscale=False,
			zmin=0,
			zmax=1,
			marker_line_color="#f8f0e1",
			marker_line_width=0.8,
		)
	)
	fig.update_geos(
		fitbounds="locations",
		visible=False,
		projection_type="mercator",
		bgcolor="rgba(0, 0, 0, 0)",
	)
	fig.update_layout(
		height=360,
		margin={"r": 0, "t": 0, "l": 0, "b": 0},
		paper_bgcolor="rgba(0, 0, 0, 0)",
		plot_bgcolor="rgba(0, 0, 0, 0)",
	)

	return fig


def add_location_markers(fig: go.Figure, resolved_locations: tuple[ResolvedAnimalLocation, ...]) -> go.Figure:
	if not resolved_locations:
		return fig

	fig.add_trace(
		go.Scattergeo(
			lon=[location.location.longitude for location in resolved_locations],
			lat=[location.location.latitude for location in resolved_locations],
			mode="markers",
			text=[location.location.area_name for location in resolved_locations],
			customdata=[
				[
					location.location.location_id,
					location.district_name or "Unknown district",
					location.province_name or "Unknown province",
					location.location.ranch_name,
				]
				for location in resolved_locations
			],
			hovertemplate=(
				"<b>%{text}</b><br>Ranch: %{customdata[3]}<br>District: %{customdata[1]}"
				"<br>Province: %{customdata[2]}<extra></extra>"
			),
			marker={
				"size": 14,
				"color": "#a86d32",
				"line": {"color": "#fffaf4", "width": 2},
				"symbol": "circle",
			},
			showlegend=False,
		)
	)
	fig.update_layout(clickmode="event+select")
	return fig


def build_sa_scatter_range_map(range_context: SpeciesRangeContext) -> go.Figure:

	highlighted_provinces = set(range_context.provinces)
	inactive_lons: list[float] = []
	inactive_lats: list[float] = []
	inactive_names: list[str] = []
	highlighted_lons: list[float] = []
	highlighted_lats: list[float] = []
	highlighted_names: list[str] = []

	for province in SA_PROVINCES:
		lat, lon = PROVINCE_CENTROIDS[province]
		if province in highlighted_provinces:
			highlighted_lons.append(lon)
			highlighted_lats.append(lat)
			highlighted_names.append(province)
			continue

		inactive_lons.append(lon)
		inactive_lats.append(lat)
		inactive_names.append(province)

	fig = go.Figure()
	fig.add_trace(
		go.Scattergeo(
			lon=inactive_lons,
			lat=inactive_lats,
			mode="markers",
			text=inactive_names,
			hovertemplate="<b>%{text}</b><br>Not highlighted for this species<extra></extra>",
			marker={
				"size": 15,
				"color": "#d8c7ae",
				"line": {"color": "#7d542f", "width": 0.8},
			},
			showlegend=False,
		)
	)
	fig.add_trace(
		go.Scattergeo(
			lon=highlighted_lons,
			lat=highlighted_lats,
			mode="markers+text",
			text=highlighted_names,
			textposition="top center",
			hovertemplate=(
				"<b>%{text}</b><br>Highlighted for "
				f"{range_context.display_name}<extra></extra>"
			),
			marker={
				"size": 22,
				"color": "#2f6d4d",
				"line": {"color": "#f8f0e1", "width": 2},
			},
			textfont={"color": "#241711", "size": 11},
			showlegend=False,
		)
	)

	fig.update_geos(
		scope="africa",
		projection_type="mercator",
		showland=True,
		landcolor="#f6efe2",
		showocean=True,
		oceancolor="#dbe9ef",
		showcoastlines=True,
		coastlinecolor="#8b745a",
		showcountries=True,
		countrycolor="#8b745a",
		showframe=False,
		bgcolor="rgba(0, 0, 0, 0)",
		lataxis_range=[-35.8, -21.5],
		lonaxis_range=[16.0, 33.5],
		resolution=50,
	)
	fig.update_layout(
		height=360,
		margin={"r": 0, "t": 0, "l": 0, "b": 0},
		paper_bgcolor="rgba(0, 0, 0, 0)",
		plot_bgcolor="rgba(0, 0, 0, 0)",
		dragmode=False,
	)

	return fig


def build_sa_range_map(species_name: str | None) -> go.Figure | None:
	range_context = get_species_range_context(species_name)
	if range_context is None:
		return None

	geojson = load_sa_geojson()
	if geojson is not None:
		return build_sa_geojson_range_map(range_context, geojson)

	return build_sa_scatter_range_map(range_context)


def build_animal_location_map(
	species_name: str | None,
	locations: tuple[AnimalLocation, ...],
) -> go.Figure | None:
	if not locations:
		return None

	geojson = load_sa_geojson()
	resolved_locations = resolve_animal_locations(locations, geojson)
	range_context = get_species_range_context(species_name)

	if geojson is not None and range_context is not None:
		fig = build_sa_geojson_range_map(range_context, geojson)
	elif geojson is not None:
		fig = build_neutral_geojson_map(geojson)
	elif range_context is not None:
		fig = build_sa_scatter_range_map(range_context)
	else:
		return None

	return add_location_markers(fig, resolved_locations)


def build_district_detail_map(location: AnimalLocation) -> go.Figure | None:
	geojson = load_sa_geojson()
	resolved_location = resolve_animal_location(location, geojson)

	if geojson is not None and resolved_location.district_name:
		selected_feature = get_geojson_features_by_name().get(resolved_location.district_name)
		if selected_feature is not None:
			locations: list[str] = []
			zone_values: list[int] = []
			hover_text: list[str] = []

			for feature in geojson.get("features", []):
				feature_name = feature.get("properties", {}).get("name")
				if not feature_name:
					continue

				if feature_name == resolved_location.district_name:
					zone_value = 2
				elif DISTRICT_TO_PROVINCE.get(feature_name) == resolved_location.province_name:
					zone_value = 1
				else:
					zone_value = 0

				locations.append(feature_name)
				zone_values.append(zone_value)
				hover_text.append(
					f"<b>{feature_name}</b><br>Province: {DISTRICT_TO_PROVINCE.get(feature_name, 'Unknown')}<extra></extra>"
				)

			fig = go.Figure(
				go.Choropleth(
					geojson=geojson,
					locations=locations,
					z=zone_values,
					featureidkey="properties.name",
					text=hover_text,
					hovertemplate="%{text}",
					colorscale=[
						[0.0, "#efe4d2"],
						[0.4999, "#efe4d2"],
						[0.5, "#d8c7ae"],
						[0.7499, "#d8c7ae"],
						[0.75, "#2f6d4d"],
						[1.0, "#2f6d4d"],
					],
					showscale=False,
					zmin=0,
					zmax=2,
					marker_line_color="#f8f0e1",
					marker_line_width=0.8,
				)
			)
			add_location_markers(fig, (resolved_location,))

			bounds = get_feature_bounds(selected_feature)
			if bounds is not None:
				min_lon, max_lon, min_lat, max_lat = bounds
				lon_padding = max((max_lon - min_lon) * 0.18, 0.22)
				lat_padding = max((max_lat - min_lat) * 0.18, 0.18)
				fig.update_geos(
					visible=False,
					projection_type="mercator",
					bgcolor="rgba(0, 0, 0, 0)",
					lonaxis_range=[min_lon - lon_padding, max_lon + lon_padding],
					lataxis_range=[min_lat - lat_padding, max_lat + lat_padding],
				)
			else:
				fig.update_geos(
					visible=False,
					projection_type="mercator",
					bgcolor="rgba(0, 0, 0, 0)",
				)

			fig.update_layout(
				height=400,
				margin={"r": 0, "t": 0, "l": 0, "b": 0},
				paper_bgcolor="rgba(0, 0, 0, 0)",
				plot_bgcolor="rgba(0, 0, 0, 0)",
			)
			return fig

	fig = go.Figure(
		go.Scattergeo(
			lon=[location.longitude],
			lat=[location.latitude],
			mode="markers",
			text=[location.area_name],
			hovertemplate="<b>%{text}</b><br>District detail fallback<extra></extra>",
			marker={"size": 16, "color": "#2f6d4d", "line": {"color": "#fffaf4", "width": 2}},
			showlegend=False,
		)
	)
	fig.update_geos(
		projection_type="mercator",
		showland=True,
		landcolor="#f6efe2",
		showocean=True,
		oceancolor="#dbe9ef",
		showcoastlines=True,
		coastlinecolor="#8b745a",
		showcountries=True,
		countrycolor="#8b745a",
		showframe=False,
		bgcolor="rgba(0, 0, 0, 0)",
		lonaxis_range=[location.longitude - 1.0, location.longitude + 1.0],
		lataxis_range=[location.latitude - 0.8, location.latitude + 0.8],
	)
	fig.update_layout(
		height=400,
		margin={"r": 0, "t": 0, "l": 0, "b": 0},
		paper_bgcolor="rgba(0, 0, 0, 0)",
		plot_bgcolor="rgba(0, 0, 0, 0)",
	)
	return fig