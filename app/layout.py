from __future__ import annotations

from urllib.parse import urlencode

from dash import dcc, html


STATIC_MAP_CONFIG = {
	"displayModeBar": False,
	"responsive": True,
	"staticPlot": True,
	"scrollZoom": False,
	"doubleClick": False,
	"editable": False,
}


def build_upload_component(upload_key: str | None = None) -> html.Div:
	return html.Div(
		id=f"upload-instance-{upload_key or 'initial'}",
		children=[
			dcc.Upload(
				id="image-upload",
				className="upload-area",
				multiple=False,
				children=html.Div(
					children=[
						html.Span("Drag, drop, or browse", className="upload-kicker"),
						html.H3("Upload one animal photo", className="upload-title"),
						html.P(
							"Accepted formats: JPG, PNG, WEBP. The animal map and full catalog remain hidden until an image has been uploaded.",
							className="upload-subtitle",
						),
					],
				),
			),
		],
	)


def metric_card(label: str, value: str) -> html.Div:
	return html.Div(
		className="metric-card",
		children=[
			html.Span(label, className="metric-label"),
			html.Span(value, className="metric-value"),
		],
	)


def meta_item(label: str, value: str) -> html.Div:
	return html.Div(
		className="meta-item",
		children=[
			html.Span(label, className="meta-label"),
			html.Span(value, className="meta-value"),
		],
	)


def detail_card(label: str, value: str) -> html.Div:
	return html.Div(
		className="detail-card",
		children=[
			html.Span(label, className="detail-label"),
			html.Span(value, className="detail-value"),
		],
	)


def location_page_href(animal_id: str, location_id: str) -> str:
	return "/districts?" + urlencode({"animal": animal_id, "location": location_id})


def location_card(animal_id: str, location, resolved_location) -> html.Article:
	district_name = resolved_location.district_name or "District pending"
	province_name = resolved_location.province_name or "Province pending"
	highlight_list = None
	if location.highlights:
		highlight_list = html.Ul(
			className="reason-list",
			children=[html.Li(highlight) for highlight in location.highlights],
		)

	return html.Article(
		className="location-card",
		children=[
			html.Span(location.ranch_name, className="location-kicker"),
			html.H3(location.area_name),
			html.Div(
				className="location-meta-row",
				children=[
					html.Span(f"{district_name}, {province_name}", className="location-meta"),
					html.Span(
						f"{location.latitude:.4f}, {location.longitude:.4f}",
						className="location-meta",
					),
				],
			),
			highlight_list,
			dcc.Link(
				"Open district view",
				href=location_page_href(animal_id, location.location_id),
				className="animal-link",
			),
		],
	)


def animal_card(animal) -> html.Article:
	return html.Article(
		className="animal-card",
		children=[
			html.Img(src=animal.image_src, alt=animal.name, className="animal-image"),
			html.Div(
				className="animal-body",
				children=[
					html.Span(animal.category, className="animal-category"),
					html.H3(animal.name),
					html.P(animal.scientific_name, className="animal-scientific"),
					html.P(animal.short_description, className="animal-copy"),
					dcc.Link("View animal map", href=animal.page_href, className="animal-link"),
				],
			),
		],
	)


def create_catalog_page(animals) -> html.Div:
	return html.Div(
		className="page",
		children=[
			html.Section(
				className="panel catalog-page-panel",
				children=[
					dcc.Link("Back to home", href="/", className="back-link"),
					html.Div(
						className="eyebrow-row",
						children=[
							html.Span("Animal catalog", className="eyebrow"),
							html.Span(f"{len(animals)} mapped animals", className="status-pill"),
						],
					),
					html.H1("Browse the full animal catalog"),
					html.P(
						"Select an animal to open its map page. Each animal page is now focused on the South Africa map and the stored ranch markers for that species.",
						className="hero-copy",
					),
					html.Div(
						className="animal-grid",
						children=[animal_card(animal) for animal in animals],
					),
				],
			),
		],
	)


def empty_preview() -> html.Div:
	return html.Div(
		className="preview-empty",
		children=[
			html.Div(
				children=[
					html.H3("Animal preview"),
					html.P(
						"Your uploaded animal image will appear here before it is checked against the Bosvelder Ranch catalog."
					),
				]
			),
		],
	)


def image_preview(contents: str, file_name: str, size_kb: float) -> html.Div:
	return html.Div(
		className="preview-card",
		children=[
			html.Img(src=contents, alt=f"Preview of {file_name}"),
			html.Div(
				className="preview-meta",
				children=[
					html.Span(file_name, className="preview-name"),
					html.Span(f"{size_kb:.1f} KB", className="preview-size"),
				],
			),
		],
	)


def error_preview(message: str) -> html.Div:
	return html.Div(
		className="preview-empty",
		children=[
			html.Div(
				children=[
					html.H3("Upload issue"),
					html.P(message),
				],
			),
		],
	)


def default_result_card() -> html.Div:
	return html.Div(
		className="result-shell",
		children=[
			html.Div(
				className="result-top",
				children=[
					html.Div(
						children=[
							html.Span("Animal image search", className="result-mode"),
							html.H2("Awaiting ranch match", className="result-heading"),
						],
					),
					html.Span("No animal match yet", className="result-badge result-badge--neutral"),
				],
			),
			html.P(
				"Upload a clear animal image to check whether the species appears on Bosvelder Ranch. If a match is found, you can open the animal profile page.",
				className="result-summary",
			),
			html.Div(
				className="confidence-card",
				children=[
					html.Div(
						className="confidence-row",
						children=[html.Span("Animal match confidence"), html.Strong("--")],
					),
					html.Div(
						className="confidence-track",
						children=[
							html.Div(className="confidence-fill confidence-fill--neutral", style={"width": "22%"}),
						],
					),
				],
			),
			html.Div(
				className="meta-grid",
				children=[
					meta_item("Catalog", "South African ranch animals"),
					meta_item("Search", "Upload an animal image"),
					meta_item("Next step", "View animal page"),
				],
			),
			html.Div(
				className="note",
				children="This demo currently uses deterministic placeholder logic. Replace it with the trained model once your ranch animal classifier is ready.",
			),
		],
	)


def error_result_card(message: str) -> html.Div:
	return html.Div(
		className="result-shell",
		children=[
			html.Div(
				className="result-top",
				children=[
					html.Div(
						children=[
							html.Span("Upload validation", className="result-mode"),
							html.H2("Action required", className="result-heading"),
						],
					),
					html.Span("Fix upload", className="result-badge result-badge--error"),
				],
			),
			html.P(message, className="result-summary"),
			html.Div(
				className="note",
				children="Use a JPG, PNG, or WEBP image and try again.",
			),
		],
	)


def range_map_card(prediction) -> html.Div | None:
	if prediction.range_map_figure is None or not prediction.range_provinces:
		return None

	return html.Div(
		className="range-card",
		children=[
			html.Div(
				className="range-header",
				children=[
					html.Span("South Africa range", className="range-eyebrow"),
					html.Div(
						className="range-chip-row",
						children=[
							html.Span(province, className="range-chip")
							for province in prediction.range_provinces
						],
					),
				],
			),
			html.P(prediction.range_summary, className="range-copy"),
			dcc.Graph(
				figure=prediction.range_map_figure,
				className="range-graph",
				config=STATIC_MAP_CONFIG,
			),
		],
	)


def result_card(prediction) -> html.Div:
	confidence_width = f"{prediction.confidence * 100:.0f}%"
	action_links: list[html.Component] = []
	range_card = range_map_card(prediction)
	if prediction.action_href and prediction.action_label:
		action_links.append(
			dcc.Link(
			prediction.action_label,
			href=prediction.action_href,
			className="result-action",
			)
		)
	if prediction.secondary_action_href and prediction.secondary_action_label:
		action_links.append(
			dcc.Link(
				prediction.secondary_action_label,
				href=prediction.secondary_action_href,
				className="result-action result-action--secondary",
			)
		)

	action_row = None
	if action_links:
		action_row = html.Div(className="action-row", children=action_links)

	return html.Div(
		className="result-shell",
		children=[
			html.Div(
				className="result-top",
				children=[
					html.Div(
						children=[
							html.Span(prediction.mode_label, className="result-mode"),
							html.H2(prediction.title, className="result-heading"),
						],
					),
					html.Span(
						prediction.badge_text,
						className=f"result-badge {prediction.badge_class}",
					),
				],
			),
			html.P(prediction.summary, className="result-summary"),
			html.Div(
				className="confidence-card",
				children=[
					html.Div(
						className="confidence-row",
						children=[
							html.Span("Animal match confidence"),
							html.Strong(confidence_width),
						],
					),
					html.Div(
						className="confidence-track",
						children=[
							html.Div(
								className=f"confidence-fill {prediction.fill_class}",
								style={"width": confidence_width},
							),
						],
					),
				],
			),
			html.Div(
				className="meta-grid",
				children=[meta_item(label, value) for label, value in prediction.details],
			),
			range_card,
			html.Ul(
				className="reason-list",
				children=[html.Li(reason) for reason in prediction.reasons],
			),
			action_row,
			html.Div(className="note", children=prediction.note),
		],
	)


def create_home_page() -> html.Div:
	return html.Div(
		className="page",
		children=[
			html.Section(
				className="panel hero",
				children=[
					html.Div(
						className="eyebrow-row",
						children=[
							html.Span("South African ranch catalog", className="eyebrow"),
							html.Span("Interactive catalog", className="status-pill"),
						],
					),
					html.H1("See which South African animals live on Bosvelder Ranch"),
					html.P(
						"Upload an animal image to unlock the next step in the experience. After the image is checked, the visitor can open the specific animal map page or move into the full catalog.",
						className="hero-copy",
					),
					html.Div(
						className="hero-metrics",
						children=[
							metric_card("Species focus", "South African wildlife"),
							metric_card("Search mode", "Upload an animal image"),
							metric_card("Unlocked after upload", "Animal map or full catalog"),
						],
					),
				],
			),
			html.Section(
				className="panel upload-panel home-upload-panel",
				children=[
					html.H2("Search by animal image", className="panel-title"),
					html.P(
						"Start by uploading an animal photo. Once the system has processed the image, the result area will reveal links to the specific animal map page and the full catalog.",
						className="panel-copy",
					),
					html.Div(id="upload-shell", children=build_upload_component()),
					html.Div(
						className="helper-row",
						children=[
							html.Span("Upload-first experience", className="helper-chip"),
							html.Span("Animal map after prediction", className="helper-chip"),
							html.Span("Full catalog after upload", className="helper-chip"),
						],
					),
				],
			),
			html.Div(id="result-panel"),
		],
	)


def create_animal_page(
	animal,
	locations=(),
	resolved_locations=(),
	location_map_figure=None,
) -> html.Div:
	map_copy = (
		"This page shows the South Africa range for the selected animal. Once hunt locations are added, the page can show static markers and district-view cards without relying on map clicks."
	)
	if locations:
		map_copy = (
			"This page shows the South Africa range for the selected animal together with the saved hunting locations. "
			"The map markers are visual only; use the location cards below to open the district view for a specific site."
		)

	location_section = None
	if locations and resolved_locations:
		location_section = html.Div(
			className="animal-page-content",
			children=[
				html.Div(
					className="eyebrow-row",
					children=[
						html.Span("Hunt locations", className="eyebrow"),
						html.Span(f"{len(locations)} district views", className="status-pill"),
					],
				),
				html.P(
					"The South Africa map stays static for lighter hosting on Render. Open a district page from one of the hunt-location cards below.",
					className="panel-copy",
				),
				html.Div(
					className="location-grid",
					children=[
						location_card(animal.animal_id, location, resolved_location)
						for location, resolved_location in zip(locations, resolved_locations)
					],
				),
			],
		)

	return html.Div(
		className="page",
		children=[
			html.Section(
				className="panel animal-map-panel",
				children=[
					dcc.Link("Back to full catalog", href="/catalog", className="back-link"),
					html.Div(
						className="eyebrow-row",
						children=[
							html.Span("Animal map", className="eyebrow"),
							html.Span(f"{len(locations)} mapped locations", className="status-pill"),
						],
					),
					html.H1(animal.name),
					html.P(
						map_copy,
						className="hero-copy",
					),
					dcc.Graph(
						id="animal-location-map",
						figure=location_map_figure,
						className="range-graph",
						config=STATIC_MAP_CONFIG,
					),
					location_section,
					(
						html.Div(
							className="note",
							children="No hunting locations have been added for this animal yet, so the page is currently showing only the broader species range map.",
						)
						if not locations
						else None
					),
				],
			),
		],
	)


def create_district_page(animal, location, resolved_location, district_map_figure) -> html.Div:
	district_name = resolved_location.district_name or "District unavailable"
	province_name = resolved_location.province_name or "Province unavailable"
	district_map = (
		dcc.Graph(
			figure=district_map_figure,
			className="range-graph",
			config=STATIC_MAP_CONFIG,
		)
		if district_map_figure is not None
		else html.Div(
			className="note",
			children="We could not build the district map for this location yet.",
		)
	)

	highlight_section = (
		html.Ul(
			className="traits-list",
			children=[html.Li(highlight) for highlight in location.highlights],
		)
		if location.highlights
		else html.P(
			"Add a short list of district highlights for this ranch location to expand the explanation on this page.",
			className="panel-copy",
		)
	)

	return html.Div(
		className="page",
		children=[
			html.Section(
				className="panel animal-page-panel",
				children=[
					dcc.Link("Back to animal page", href=animal.page_href, className="back-link"),
					html.Div(
						className="eyebrow-row",
						children=[
							html.Span("District view", className="eyebrow"),
							html.Span(district_name, className="status-pill"),
						],
					),
					html.H1(location.area_name),
					html.P(location.district_story, className="hero-copy"),
					html.Div(
						className="district-layout-grid",
						children=[
							html.Div(className="district-map-shell", children=[district_map]),
							html.Div(
								className="animal-page-content",
								children=[
									html.Div(
										className="store-detail-grid",
										children=[
											detail_card("Animal", animal.name),
											detail_card("Listing", location.ranch_name),
											detail_card("District", district_name),
											detail_card("Province", province_name),
											detail_card(
												"Coordinates",
												f"{location.latitude:.4f}, {location.longitude:.4f}",
											),
										],
									),
									html.H3("Area highlights", className="detail-heading"),
									highlight_section,
								],
							),
						],
					),
				],
			),
		],
	)


def create_district_missing_page() -> html.Div:
	return html.Div(
		className="page",
		children=[
			html.Section(
				className="panel animal-page-panel",
				children=[
					dcc.Link("Back to animal catalog", href="/catalog", className="back-link"),
					html.Div(
						className="eyebrow-row",
						children=[
							html.Span("District view", className="eyebrow"),
							html.Span("Unavailable", className="status-pill"),
						],
					),
					html.H1("District page not available"),
					html.P(
						"We could not load the requested hunt location. Return to the animal page and choose another mapped area.",
						className="hero-copy",
					),
				],
			),
		],
	)


def create_animal_missing_page() -> html.Div:
	return html.Div(
		className="page",
		children=[
			html.Section(
				className="panel animal-page-panel",
				children=[
					dcc.Link("Back to full catalog", href="/catalog", className="back-link"),
					html.Div(
						className="eyebrow-row",
						children=[
							html.Span("Animal profile", className="eyebrow"),
							html.Span("Not available", className="status-pill"),
						],
					),
					html.H1("Animal map not available"),
					html.P(
						"We could not load an animal map for this selection. Return to the catalog and try another animal.",
						className="hero-copy",
					),
				],
			),
		],
	)


def create_layout() -> html.Div:
	return html.Div(
		className="shell",
		children=[
			dcc.Location(id="url"),
			html.Div(className="ambient ambient-a"),
			html.Div(className="ambient ambient-b"),
			html.Div(id="page-content"),
		],
	)
