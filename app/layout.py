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


def empty_preview() -> html.Div:
	return html.Div(
		className="preview-empty",
		children=[
			html.Div(
				children=[
					html.H3("Image preview"),
					html.P(
						"Your uploaded product image will appear here so you can "
						"demo the full user journey before the model is connected."
					),
				]
			)
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
				]
			)
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
							html.Span("Prototype mode", className="result-mode"),
							html.H2("Awaiting upload", className="result-heading"),
						]
					),
					html.Span("No decision yet", className="result-badge result-badge--neutral"),
				],
			),
			html.P(
				"Upload a product image to preview the classification experience. "
				"The interface is production-shaped, while the decision logic is still mocked.",
				className="result-summary",
			),
			html.Div(
				className="confidence-card",
				children=[
					html.Div(
						className="confidence-row",
						children=[html.Span("Prototype confidence"), html.Strong("--")],
					),
					html.Div(
						className="confidence-track",
						children=[html.Div(className="confidence-fill confidence-fill--neutral", style={"width": "22%"})],
					),
				],
			),
			html.Div(
				className="meta-grid",
				children=[
					meta_item("Deployment", "Render-ready"),
					meta_item("Classifier", "Demo logic"),
					meta_item("Swap point", "1 function"),
				],
			),
			html.Div(
				className="note",
				children=(
					"When your trained model is ready, replace the placeholder logic in "
					"run_model_prediction() and keep the rest of this app as-is."
				),
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
						]
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


def result_card(prediction, file_name: str, file_type: str, size_kb: float) -> html.Div:
	confidence_width = f"{prediction.confidence * 100:.0f}%"
	return html.Div(
		className="result-shell",
		children=[
			html.Div(
				className="result-top",
				children=[
					html.Div(
						children=[
							html.Span(prediction.mode_label, className="result-mode"),
							html.H2(prediction.label, className="result-heading"),
						]
					),
					html.Span(
						prediction.label,
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
							html.Span("Prototype confidence"),
							html.Strong(confidence_width),
						],
					),
					html.Div(
						className="confidence-track",
						children=[
							html.Div(
								className=f"confidence-fill {prediction.fill_class}",
								style={"width": confidence_width},
							)
						],
					),
				],
			),
			html.Div(
				className="meta-grid",
				children=[
					meta_item("File", file_name),
					meta_item("Type", file_type),
					meta_item("Size", f"{size_kb:.1f} KB"),
				],
			),
			html.Ul(
				className="reason-list",
				children=[html.Li(reason) for reason in prediction.reasons],
			),
			html.Div(
				className="note",
				children=(
					"This result is safe for demos, UI reviews, and deployment previews. "
					"Replace the mocked decision with real model inference later."
				),
			),
		],
	)


def create_layout() -> html.Div:
	return html.Div(
		className="shell",
		children=[
			html.Div(className="ambient ambient-a"),
			html.Div(className="ambient ambient-b"),
			html.Div(
				className="page",
				children=[
					html.Section(
						className="panel hero",
						children=[
							html.Div(
								className="eyebrow-row",
								children=[
									html.Span("Render-ready Dash prototype", className="eyebrow"),
									html.Span("Model pending", className="status-pill"),
								],
							),
							html.H1("Catalog image screening for your storefront"),
							html.P(
								"Upload a product image to test the experience, present the concept, "
								"and deploy the interface now while your machine learning model is still in progress.",
								className="hero-copy",
							),
							html.Div(
								className="hero-metrics",
								children=[
									metric_card("Decision", "Sold / Not sold"),
									metric_card("Deployment", "Render"),
									metric_card("Model state", "Prototype logic"),
								],
							),
						],
					),
					html.Div(
						className="main-grid",
						children=[
							html.Section(
								className="panel upload-panel",
								children=[
									html.H2("Upload a product image", className="panel-title"),
									html.P(
										"This version is designed for demos and stakeholder reviews. "
										"It shows the full interaction pattern without pretending that a trained model already exists.",
										className="panel-copy",
									),
									dcc.Upload(
										id="image-upload",
										className="upload-area",
										multiple=False,
										children=html.Div(
											children=[
												html.Span("Drag, drop, or browse", className="upload-kicker"),
												html.H3("Drop in one product photo", className="upload-title"),
												html.P(
													"Accepted formats: JPG, PNG, WEBP. The image will be previewed immediately.",
													className="upload-subtitle",
												),
											],
										),
									),
									html.Div(
										className="helper-row",
										children=[
											html.Span("Single upload flow", className="helper-chip"),
											html.Span("Deterministic demo result", className="helper-chip"),
											html.Span("Ready to deploy on Render", className="helper-chip"),
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
							html.H2("How to move from prototype to real inference", className="panel-title"),
							html.P(
								"The frontend is already ready for deployment. When your model is finished, you only need to swap the classifier function.",
								className="panel-copy",
							),
							html.Div(
								className="roadmap-grid",
								children=[
									roadmap_card(
										"1",
										"Train and save the model",
										"Export the final artifact once your image classifier can decide whether an uploaded product belongs to your store catalog.",
									),
									roadmap_card(
										"2",
										"Wire the inference function",
										"Replace run_model_prediction() in src/web_app.py so the upload bytes are sent through the real model instead of the demo rules.",
									),
									roadmap_card(
										"3",
										"Deploy without redesign",
										"Keep the same Dash layout, Render config, and user flow. Only the decision engine changes.",
									),
								],
							),
						],
					),
				],
			),
		],
	)