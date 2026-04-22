# E-Commerce Image Classification

This project is a Dash-based e-commerce prototype for animal hunting sales. It combines a sale catalog, animal detail pages, South African range and location maps, and image-based animal identification so a user can upload a photo and see whether the species is part of the current sold list.

The website now uses a single MobileNetV2 TensorFlow Lite runtime through LiteRT. Training and notebook workflows remain separate TensorFlow workflows and are not loaded by the website at request time.

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
- `src/model_runtime.py`: TFLite website runtime plus the MobileNetV2 asset export command.
- `src/train_models.py`: fine-tuning workflow for project models.
- `src/generate_map_images.py`: static map export pipeline.
- `app/`: layout, styling, catalog data, and map helpers.
- `artifacts/`: charts, metadata, and model-related deployment assets.
- `data/`: raw datasets and image files.
- `notebooks/`: EDA, map generation, comparisons, and SHAP analysis.

## Requirements Files

- `requirements.txt`: local app runtime plus the packages needed to export the TFLite assets.
- `Installable-requirements.txt`: broader package snapshot for notebook, training, and analysis workflows.
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

## Website Runtime

The website now has one prediction path.

- Runtime: MobileNetV2 TensorFlow Lite through LiteRT
- Assets: `artifacts/mobilenet_v2_imagenet.tflite` and `artifacts/imagenet_class_index.json`
- Result logic: the TFLite model predicts the closest ImageNet label, and the app maps matching species into the sold catalog
- Fallback: if the LiteRT output does not map to a sold class, the site can still use filename cues before routing the upload to not sold

This keeps the hosted app lightweight while leaving training scripts separate from the live request path.

## Export LiteRT Assets for Render

Render uses a TensorFlow Lite version of MobileNetV2 together with a checked-in ImageNet label index.

Generate those assets locally with:

```bash
python src/model_runtime.py
```

This command writes:

- `artifacts/mobilenet_v2_imagenet.tflite`
- `artifacts/imagenet_class_index.json`

TensorFlow and Keras are only needed for this export step. The hosted website runtime itself uses LiteRT.

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

Render notes:

- The hosted site now uses LiteRT instead of the full TensorFlow package.
- This keeps the deployment lighter while preserving image-driven predictions.
- The hosted path depends on `artifacts/mobilenet_v2_imagenet.tflite` and `artifacts/imagenet_class_index.json` being present in the repo.

## Training and Analysis

Training and notebook workflows remain separate TensorFlow workflows.

Use an environment with TensorFlow installed before running those scripts or regenerating the TFLite export assets.

Useful files:

- `src/train_models.py`
- `src/prepare_data.py`
- `src/pre_process_data.py`
- `src/SHAP_analysis.py`
- `notebooks/Model Comparisons.ipynb`
- `notebooks/SHAP_analysis.ipynb`

## Current State and Limitations

- The website uses a generic MobileNetV2/ImageNet TFLite model and a strict sold-animal mapping, so it is still not a replacement for a properly fine-tuned project classifier.
- File-name fallback logic still exists for cases where a generic label does not cleanly map to a sold class.
- Render is optimized for inference and static asset serving, not for model training.

## Summary

Use `requirements.txt` for the web app and TFLite asset export, `Installable-requirements.txt` when you want the broader ML workflow dependencies, and `requirements-render.txt` for the hosted Render service. For Render, export the LiteRT assets, keep the static maps committed, and deploy without any model-backend flags.
