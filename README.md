# E-Commerce Image Classification

This project is a Dash-based e-commerce prototype for animal hunting sales. It combines a sale catalog, animal detail pages, South African range and location maps, and image-based animal identification so a user can upload a photo and see whether the species is part of the current sold list.

The application is published on Render and is designed around two runtime modes:

- Local development and model training can still use full TensorFlow and Keras.
- Render deployment now uses TensorFlow Lite through LiteRT so the hosted web app does not need the full TensorFlow runtime.

## What the App Does

- Shows a catalog of animals currently listed for sale.
- Displays animal pages, district pages, and species range maps.
- Accepts JPG, PNG, and WEBP uploads.
- Classifies uploads into sold or not sold.
- Routes anything outside the configured sale list into the not sold result.
- Uses pre-generated static map images so Render does not need to build Plotly maps at request time.

The sold-animal catalog currently maps to these ten configured sale species:

- Kudu
- Springbok
- Giraffe
- Buffalo
- Rhino
- Zebra
- Ostrich
- Elephant
- Lion
- Hippopotamus

## Project Layout

- `src/web_app.py`: Dash entrypoint used by Gunicorn on Render.
- `src/model_runtime.py`: runtime prediction logic for TensorFlow and LiteRT backends.
- `src/train_models.py`: fine-tuning workflow for project models.
- `src/generate_map_images.py`: static map export pipeline.
- `app/export_tflite_assets.py`: exports the MobileNetV2 LiteRT model and ImageNet label index for Render.
- `app/`: layout, styling, catalog data, and map helpers.
- `artifacts/`: charts, metadata, and model-related deployment assets.
- `data/`: raw datasets and image files.
- `notebooks/`: EDA, map generation, comparisons, and SHAP analysis.

## Requirements Files

- `requirements.txt`: local app runtime with full TensorFlow for development.
- `requirements-ml.txt`: training, preprocessing, visualization, and analysis dependencies.
- `requirements-render.txt`: Render deployment dependencies using LiteRT instead of TensorFlow.

## Local Setup

1. Create and activate a virtual environment.
2. Install the local app dependencies.

```bash
pip install -r requirements.txt
```

3. Start the web app.

```bash
python src/web_app.py
```

The deployed server entrypoint is:

```bash
gunicorn src.web_app:server --workers 1 --threads 2 --timeout 180
```

## Image Classification Modes

The app supports two prediction paths.

### 1. Generic ImageNet Runtime

This is the safest hosted option and the current Render default.

- Local backend: `mobilenet_v2`, `mobilenet_v3_small`, `efficientnet_b0`, or `resnet50`
- Render backend: `mobilenet_v2_tflite`
- Result logic: a generic vision model predicts the closest ImageNet label, and the app maps matching species into the sold catalog

### 2. Fine-Tuned Project Model

This path is intended for a real project-specific classifier trained from your dataset.

- Backend: `project_model`
- Artifact metadata: `artifacts/best_fine_tuned_model.json`
- Model file expected locally: a matching `.h5` file in `artifacts/`

Important:

- The checked-in metadata currently only lists four classes: buffalo, elephant, rhino, and zebra.
- Because the configured sale catalog contains ten sold species, the strict project-model path is intentionally blocked unless the metadata and model cover the full configured catalog.
- That means the current hosted app should use the generic MobileNetV2 LiteRT path unless you add a complete fine-tuned artifact set.

## Export LiteRT Assets for Render

Render uses a TensorFlow Lite version of MobileNetV2 together with a checked-in ImageNet label index.

Generate those assets locally with:

```bash
python app/export_tflite_assets.py
```

This script writes:

- `artifacts/mobilenet_v2_imagenet.tflite`
- `artifacts/imagenet_class_index.json`

These files are required for the `mobilenet_v2_tflite` backend.

## Generate Static Map Images

To avoid expensive runtime map rendering on Render, generate the static map assets ahead of time.

1. Install the extra ML and plotting dependencies.

```bash
pip install -r requirements-ml.txt
```

2. Generate the map images.

```bash
python src/generate_map_images.py
```

This produces static images for:

- Species range maps
- Animal pages
- District pages

If a generated map image is missing, the app shows an unavailable state instead of building the map live.

## Render Deployment

The project includes a `render.yaml`, but you can also create the service manually in the Render dashboard.

Use these settings:

- Runtime: `Python`
- Plan: `Free`
- Build Command: `pip install -r requirements-render.txt`
- Start Command: `gunicorn src.web_app:server --workers 1 --threads 2 --timeout 180`

Set these environment variables:

- `PYTHON_VERSION=3.12.10`
- `USE_TRAINED_MODEL=true`
- `MODEL_BACKEND=mobilenet_v2_tflite`

Render notes:

- The hosted site now uses LiteRT instead of the full TensorFlow package.
- This keeps the deployment lighter while preserving image-driven predictions.
- The hosted path depends on `artifacts/mobilenet_v2_imagenet.tflite` and `artifacts/imagenet_class_index.json` being present in the repo.
- If you later want to host a project-specific classifier, export and ship a matching deployment-ready artifact set first.

## Training and Analysis

Training and notebook workflows remain local TensorFlow workflows.

Useful files:

- `src/train_models.py`
- `src/prepare_data.py`
- `src/pre_process_data.py`
- `src/SHAP_analysis.py`
- `notebooks/Model Comparisons.ipynb`
- `notebooks/SHAP_analysis.ipynb`

## Current State and Limitations

- The current checked-in fine-tuned metadata is not yet a full ten-species sale model.
- The generic ImageNet runtime is useful for deployment and demo classification, but it is not a replacement for a properly trained project classifier.
- File-name fallback logic still exists for cases where a generic label does not cleanly map to a sold class.
- Render is optimized for inference and static asset serving, not for model training.

## Summary

Use `requirements.txt` for local development, `requirements-ml.txt` for training and analysis, and `requirements-render.txt` for the hosted Render service. For Render, export the LiteRT assets, keep the static maps committed, and deploy with `MODEL_BACKEND=mobilenet_v2_tflite`.
