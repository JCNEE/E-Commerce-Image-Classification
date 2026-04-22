"""Dash component builders for catalog pages, results, and static map views."""

from __future__ import annotations

from urllib.parse import urlencode

from dash import dcc, html
import plotly.graph_objects as go


def build_upload_component(upload_key: str | None = None) -> html.Div:
	"""Build the upload widget shell, optionally forcing a fresh Dash component key."""

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
	"""Render a compact metric tile used in hero and detail summaries."""

	return html.Div(
		className="metric-card",
		children=[
			html.Span(label, className="metric-label"),
			html.Span(value, className="metric-value"),
		],
	)


def meta_item(label: str, value: str) -> html.Div:
	"""Render a key-value item for prediction metadata grids."""

	return html.Div(
		className="meta-item",
		children=[
			html.Span(label, className="meta-label"),
			html.Span(value, className="meta-value"),
		],
	)


def animal_info_row(label: str, value: str) -> html.Div:
	"""Render a two-column info row for catalog animal cards."""

	return html.Div(
		className="animal-info-row",
		children=[
			html.Span(label, className="animal-info-label"),
			html.Span(value, className="animal-info-value"),
		],
	)


def detail_card(label: str, value: str) -> html.Div:
	"""Render a detail tile for animal and district pages."""

	return html.Div(
		className="detail-card",
		children=[
			html.Span(label, className="detail-label"),
			html.Span(value, className="detail-value"),
		],
	)


def location_page_href(animal_id: str, location_id: str) -> str:
	"""Build the routed district URL for a selected animal location."""

	return "/districts?" + urlencode({"animal": animal_id, "location": location_id})


def location_card(animal_id: str, location, resolved_location) -> html.Article:
	"""Render a hunt-location card that links to the district detail page."""

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
	"""Render one catalog card for the browse-all animals page."""

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
					html.Div(
						className="animal-info-stack",
						children=[
							animal_info_row("Guide price", animal.price_display),
							animal_info_row("Permit", animal.permit_status),
							animal_info_row("Safari / Game Drive", animal.safari_drive_opportunity),
						],
					),
					dcc.Link("View animal map", href=animal.page_href, className="animal-link"),
				],
			),
		],
	)


def create_catalog_page(animals) -> html.Div:
	"""Build the catalog landing page that lists every sold animal."""

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
						"Select an animal to open its map page. Guide rates are shown in South African Rand, permit guidance is surfaced per species, and every card now includes a safari or game-drive viewing option.",
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
	"""Render the placeholder shown before a user uploads an image."""

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
	"""Render the uploaded image preview and its basic file metadata."""

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
	"""Render a preview placeholder that explains why an upload was rejected."""

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
	"""Render the neutral result state shown before a classification happens."""

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
	"""Render the error state used when upload validation fails."""

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


def static_map_media(
	image_src: str | None,
	alt_text: str,
	empty_message: str,
) -> html.Div:
	"""Render a static map image when available, or a note when it is missing."""

	if image_src:
		return html.Div(
			className="range-graph",
			children=[html.Img(src=image_src, alt=alt_text, className="range-map-image")],
		)

	return html.Div(className="note", children=empty_message)


def range_map_card(prediction) -> html.Div | None:
	"""Build the province-range card for sold or recognised outside-catalog species."""

	if prediction.range_map_image_src is None or not prediction.range_provinces:
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
			static_map_media(
				image_src=prediction.range_map_image_src,
				alt_text=f"South Africa range map for {prediction.title}",
				empty_message="The range map is not available for this animal yet.",
			),
		],
	)


def build_top_candidates_figure(prediction):
	"""Create the horizontal bar chart for the strongest model candidates."""

	labels = [label for label, _ in prediction.top_candidates]
	values = [score for _, score in prediction.top_candidates]
	if not labels or not values:
		return None

	axis_max = max(0.18, min(1.0, max(values) * 1.15))
	bar_colors = ["#a86d32"] + ["rgba(168, 109, 50, 0.58)"] * max(0, len(values) - 1)
	figure = go.Figure(
		go.Bar(
			x=list(reversed(values)),
			y=list(reversed(labels)),
			orientation="h",
			marker={
				"color": list(reversed(bar_colors)),
				"line": {"color": "rgba(90, 56, 29, 0.18)", "width": 1.2},
			},
			hovertemplate="%{y}: %{x:.1%}<extra></extra>",
		)
	)
	figure.update_layout(
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
		margin={"l": 0, "r": 12, "t": 8, "b": 0},
		height=280,
		xaxis={
			"title": None,
			"range": [0, axis_max],
			"tickformat": ".0%",
			"gridcolor": "rgba(90, 56, 29, 0.10)",
			"zeroline": False,
		},
		yaxis={"title": None, "tickfont": {"size": 12}, "automargin": True},
		font={"family": '"Trebuchet MS", "Gill Sans", sans-serif', "color": "#685446"},
	)
	return figure


def build_catalog_breakdown_figure(prediction):
	"""Create the donut chart that compares sold-catalog and outside-catalog evidence."""

	labels = [label for label, _ in prediction.catalog_breakdown]
	values = [score for _, score in prediction.catalog_breakdown]
	if not labels or not values:
		return None

	color_map = {
		"Sold catalog cues": "#2f6d4d",
		"Outside catalog cues": "#9b4d2c",
		"Unresolved cues": "#76695e",
	}
	figure = go.Figure(
		go.Pie(
			labels=labels,
			values=values,
			hole=0.64,
			sort=False,
			marker={
				"colors": [color_map.get(label, "#76695e") for label in labels],
				"line": {"color": "rgba(255, 251, 243, 0.95)", "width": 2},
			},
			textinfo="percent",
			hovertemplate="%{label}: %{value:.1%}<extra></extra>",
		)
	)
	figure.update_layout(
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
		margin={"l": 0, "r": 0, "t": 8, "b": 0},
		height=280,
		showlegend=True,
		legend={"orientation": "h", "x": 0, "y": -0.08},
		annotations=[
			{
				"text": prediction.badge_text,
				"x": 0.5,
				"y": 0.5,
				"showarrow": False,
				"font": {"family": '"Palatino Linotype", "Book Antiqua", Georgia, serif', "size": 18, "color": "#241711"},
			}
		],
		font={"family": '"Trebuchet MS", "Gill Sans", sans-serif', "color": "#685446"},
	)
	return figure


def static_plotly_graph(figure) -> dcc.Graph:
	"""Render a Plotly figure in non-interactive mode for lightweight hosting."""

	return dcc.Graph(
		figure=figure,
		config={
			"displayModeBar": False,
			"staticPlot": True,
			"responsive": True,
		},
		className="classification-graph",
	)


def classification_graphs(prediction) -> html.Div | None:
	"""Build the optional classification-analytics section shown on result cards."""

	top_candidates_figure = build_top_candidates_figure(prediction)
	catalog_breakdown_figure = build_catalog_breakdown_figure(prediction)
	if top_candidates_figure is None and catalog_breakdown_figure is None:
		return None

	graph_cards: list[html.Component] = []
	if top_candidates_figure is not None:
		graph_cards.append(
			html.Div(
				className="classification-graph-card",
				children=[
					html.Span("Live model view", className="range-eyebrow"),
					html.H3("Top model candidates", className="graph-title"),
					html.P(
						"These are the strongest labels returned by the current TFLite classification run for this upload.",
						className="graph-copy",
					),
					html.Div(
						className="classification-graph-shell",
						children=[static_plotly_graph(top_candidates_figure)],
					),
				],
			)
		)

	if catalog_breakdown_figure is not None:
		graph_cards.append(
			html.Div(
				className="classification-graph-card",
				children=[
					html.Span("Catalog decision", className="range-eyebrow"),
					html.H3("Sold vs outside cues", className="graph-title"),
					html.P(
						"This compares how much of the visible top-model evidence aligns with the sale catalog versus labels outside it.",
						className="graph-copy",
					),
					html.Div(
						className="classification-graph-shell",
						children=[static_plotly_graph(catalog_breakdown_figure)],
					),
				],
			)
		)

	return html.Div(className="classification-graph-grid", children=graph_cards)


def result_card(prediction) -> html.Div:
	"""Render the full prediction result card, including actions and optional charts."""

	confidence_width = f"{prediction.confidence * 100:.0f}%"
	action_links: list[html.Component] = []
	range_card = range_map_card(prediction)
	graph_section = classification_graphs(prediction)
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
	note_block = html.Div(className="note", children=prediction.note) if prediction.note else None

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
			graph_section,
			range_card,
			html.Ul(
				className="reason-list",
				children=[html.Li(reason) for reason in prediction.reasons],
			),
			action_row,
			note_block,
		],
	)


def create_home_page() -> html.Div:
	"""Build the landing page with the upload prompt and deferred results panel."""

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
				],
			),
			html.Div(id="result-panel"),
		],
	)


def create_animal_page(
	animal,
	locations=(),
	resolved_locations=(),
	location_map_image_src=None,
) -> html.Div:
	"""Build the animal detail page with static maps, facts, and hunt-location cards."""

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

	animal_overview = html.Div(
		className="animal-page-content",
		children=[
			html.Div(
				className="store-detail-grid",
				children=[
					detail_card("Guide price", animal.price_display),
					detail_card("Permit", animal.permit_status),
					detail_card("Best viewing", animal.best_viewing),
					detail_card("Habitat zone", animal.habitat_zone),
				],
			),
			html.P(animal.description, className="animal-detail-copy"),
			html.Div(
				className="animal-feature-block",
				children=[
					html.H3("Safari / game drive opportunity", className="detail-heading"),
					html.P(animal.safari_drive_opportunity, className="animal-detail-copy"),
				],
			),
			html.Div(
				className="animal-feature-block",
				children=[
					html.H3("Permit guidance", className="detail-heading"),
					html.P(animal.permit_guidance, className="animal-detail-copy"),
				],
			),
			html.Div(
				className="animal-feature-block",
				children=[
					html.H3("Field notes", className="detail-heading"),
					html.Ul(
						className="traits-list",
						children=[html.Li(trait) for trait in animal.traits],
					),
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
					animal_overview,
					static_map_media(
						image_src=location_map_image_src,
						alt_text=f"Location map for {animal.name}",
						empty_message="No map is available for this animal yet.",
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


def create_district_page(animal, location, resolved_location, district_map_image_src) -> html.Div:
	"""Build the district detail page for one saved hunt location."""

	district_name = resolved_location.district_name or "District unavailable"
	province_name = resolved_location.province_name or "Province unavailable"
	district_map = static_map_media(
		image_src=district_map_image_src,
		alt_text=f"District map for {location.area_name}",
		empty_message="We could not build the district map for this location yet.",
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
	"""Build the fallback page shown when a district route cannot be resolved."""

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
	"""Build the fallback page shown when an animal route cannot be resolved."""

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
	"""Build the top-level shell that wraps ambient background layers and routed content."""

	return html.Div(
		className="shell",
		children=[
			dcc.Location(id="url"),
			html.Div(className="ambient ambient-a"),
			html.Div(className="ambient ambient-b"),
			html.Div(id="page-content"),
		],
	)
