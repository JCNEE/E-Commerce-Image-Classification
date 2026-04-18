from __future__ import annotations

from dash import dcc, html


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


def roadmap_card(step: str, title: str, copy: str) -> html.Div:
	return html.Div(
		className="roadmap-card",
		children=[
			html.Span(step, className="roadmap-step"),
			html.H3(title),
			html.P(copy),
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
					dcc.Link("View animal page", href=animal.page_href, className="animal-link"),
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


def result_card(prediction) -> html.Div:
	confidence_width = f"{prediction.confidence * 100:.0f}%"
	action_link = None
	if prediction.action_href and prediction.action_label:
		action_link = dcc.Link(
			prediction.action_label,
			href=prediction.action_href,
			className="result-action",
		)

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
			html.Ul(
				className="reason-list",
				children=[html.Li(reason) for reason in prediction.reasons],
			),
			action_link,
			html.Div(className="note", children=prediction.note),
		],
	)


def create_home_page(animals) -> html.Div:
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
						"Visitors can browse the ranch catalog online, learn about each species, and upload an image to check whether the animal is part of the current Bosvelder Ranch lineup.",
						className="hero-copy",
					),
					html.Div(
						className="hero-metrics",
						children=[
							metric_card("Species focus", "South African wildlife"),
							metric_card("Search mode", "Upload an animal image"),
							metric_card("Next step", "Open the animal profile"),
						],
					),
				],
			),
			html.Section(
				className="panel catalog-panel",
				children=[
					html.H2("Animals currently on the ranch", className="panel-title"),
					html.P(
						"Each profile below now uses a local animal photo together with a short explanation so visitors can compare real sightings against the ranch catalog.",
						className="panel-copy",
					),
					html.Div(
						className="animal-grid",
						children=[animal_card(animal) for animal in animals],
					),
				],
			),
			html.Div(
				className="main-grid",
				children=[
					html.Section(
						className="panel upload-panel",
						children=[
							html.H2("Search by animal image", className="panel-title"),
							html.P(
								"Upload a photo of an animal to check whether it matches one of the species currently listed for Bosvelder Ranch. Positive results link to the relevant animal page.",
								className="panel-copy",
							),
							dcc.Upload(
								id="image-upload",
								className="upload-area",
								multiple=False,
								children=html.Div(
									children=[
										html.Span("Drag, drop, or browse", className="upload-kicker"),
										html.H3("Upload one animal photo", className="upload-title"),
										html.P(
											"Accepted formats: JPG, PNG, WEBP. The image is checked against the ranch animal catalog.",
											className="upload-subtitle",
										),
									],
								),
							),
							html.Div(
								className="helper-row",
								children=[
									html.Span("South African species only", className="helper-chip"),
									html.Span("Animal explanations included", className="helper-chip"),
									html.Span("Match to animal page", className="helper-chip"),
								],
							),
							html.Div(id="image-preview", children=empty_preview()),
						],
					),
					html.Section(
						id="result-panel",
						className="panel result-panel",
						children=default_result_card(),
					),
				],
			),
			html.Section(
				className="panel roadmap",
				children=[
					html.H2("How the ranch image search works", className="panel-title"),
					html.P(
						"The catalog is built for browsing first and search second: visitors can explore the ranch species, upload an image, and then jump to the matching animal profile if the classifier recognizes it.",
						className="panel-copy",
					),
					html.Div(
						className="roadmap-grid",
						children=[
							roadmap_card(
								"1",
								"Browse the ranch catalog",
								"Visitors can see which South African animals are currently represented on the ranch and read a short explanation for each one.",
							),
							roadmap_card(
								"2",
								"Search using an image",
								"The user uploads an animal photo and the model checks whether the species appears in the Bosvelder Ranch catalog.",
							),
							roadmap_card(
								"3",
								"Open the animal profile",
								"If the animal exists on the ranch, the result card links through to a dedicated profile with habitat, viewing, and explanation details.",
							),
						],
					),
				],
			),
		],
	)


def create_animal_page(animal) -> html.Div:
	return html.Div(
		className="page",
		children=[
			html.Section(
				className="panel animal-page-panel",
				children=[
					dcc.Link("Back to ranch catalog", href="/", className="back-link"),
					html.Div(
						className="eyebrow-row",
						children=[
							html.Span("Animal profile", className="eyebrow"),
							html.Span("Currently on the ranch", className="status-pill"),
						],
					),
					html.H1(animal.name),
					html.P(animal.short_description, className="hero-copy"),
					html.Div(
						className="animal-page-grid",
						children=[
							html.Img(src=animal.image_src, alt=animal.name, className="animal-page-image"),
							html.Div(
								className="animal-page-content",
								children=[
									html.P(animal.description, className="animal-detail-copy"),
									html.Div(
										className="store-detail-grid",
										children=[
											detail_card("Scientific name", animal.scientific_name),
											detail_card("Habitat zone", animal.habitat_zone),
											detail_card("Best viewing", animal.best_viewing),
										],
									),
									html.H3("What to look for", className="detail-heading"),
									html.Ul(
										className="traits-list",
										children=[html.Li(trait) for trait in animal.traits],
									),
								],
							),
						],
					),
					html.Div(
						className="note",
						children="This profile is ready for your real ranch data. You can extend it later with live sighting updates, enclosure information, or richer animal metadata.",
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
					dcc.Link("Back to ranch catalog", href="/", className="back-link"),
					html.Div(
						className="eyebrow-row",
						children=[
							html.Span("Animal profile", className="eyebrow"),
							html.Span("Not available", className="status-pill"),
						],
					),
					html.H1("Animal page not available"),
					html.P(
						"We could not load a ranch animal profile for this selection. Return to the catalog and try another animal image.",
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
