# E-Commerce Image Classification

This repository is a machine-learning project plus a Dash web application for animal image classification and a South African sale-catalog prototype. It combines dataset preparation, transfer learning, SHAP explainability, static map generation, a lightweight inference runtime, and a deployable web interface.

The project has two connected but different tracks:

1. A research and training track that preprocesses the image dataset, trains fine-tuned CNN models, compares them, and explains predictions with SHAP.
2. A deployment track that serves a Dash app with a LiteRT/TFLite MobileNetV2 runtime, pre-generated static maps, and a sold versus not-sold catalog experience.

## Project Flow

The intended project flow is:

1. `src/prepare_data.py`
2. `src/pre_process_data.py`
3. `src/train_models.py`
4. `src/SHAP_analysis.py`
5. `src/generate_map_images.py`
6. `src/model_runtime.py`
7. `src/web_app.py`

That order is documented below, together with the notebooks that support exploration and reporting.

## What The Project Does

- Loads and preprocesses wildlife images from the project dataset.
- Trains and fine-tunes MobileNetV2, ResNet50, and EfficientNetB0.
- Saves reusable training artifacts such as models, metrics, plots, and summaries.
- Produces SHAP visual explanations for model predictions.
- Generates static South African range and district map images for the app.
- Exports a lightweight MobileNetV2 TFLite model and ImageNet label index for deployment.
- Runs a Dash web app that lets a user upload an image and view sold or not-sold catalog results.

## Key Directories

- `src/`: executable scripts for preprocessing, training, explainability, runtime export, and the web app.
- `app/`: Dash layout code, styles, animal catalog data, range logic, and generated/static UI assets.
- `artifacts/`: saved models, summaries, comparison outputs, SHAP figures, and runtime assets.
- `data/`: project CSV files and the image dataset used during preprocessing and training.
- `notebooks/`: exploratory analysis, map generation, model comparison, and SHAP notebook workflows.

## Requirements Files

- `requirements.txt`: local web app runtime plus the packages needed to export the TFLite runtime assets.
- `Installable-requirements.txt`: broader experimentation environment for training, notebooks, and analysis.
- `requirements-render.txt`: minimal Render deployment dependencies using LiteRT instead of full TensorFlow.

## Local Setup

This project is currently configured around Python 3.12.

### Create a virtual environment

Windows example:

```bash
py -3.12 -m venv venv
venv\Scripts\activate
```

If you only want to run the web app locally:

```bash
pip install -r requirements.txt
```

If you want the broader notebook and training environment:

```bash
pip install -r Installable-requirements.txt
```

If you plan to export Plotly figures as PNG files for the map workflow, make sure `kaleido` is available in the environment:

```bash
pip install kaleido
```

## End-To-End Script Guide

### 1. `src/prepare_data.py`

Purpose:
Preprocesses a single image for MobileNetV2 by resizing it to `224x224`, converting it to an array, applying `mobilenet_v2.preprocess_input`, and expanding it into batch form.

Use it when:

- You need a lightweight preprocessing helper for MobileNetV2 experiments.
- You want a reusable utility for single-image inference preparation.

Important note:
This file is a helper utility. It does not create the training dataset artifact used by `train_models.py`.

### 2. `src/pre_process_data.py`

Purpose:
Loads the raw image dataset from `data/Image_Dataset`, extracts labels from filenames, resizes images to `224x224`, normalizes pixel values to `[0, 1]`, encodes classes, creates train/validation/test splits, previews augmented training samples, and saves the processed dataset to HDF5.

Main input:

- `data/Image_Dataset`

Current saved training classes:

- `buffalo`
- `elephant`
- `rhino`
- `zebra`

Main output:

- `artifacts/wildlife_data.h5`

Run it with:

```bash
python src/pre_process_data.py
```

What it saves:

- `X_train`, `y_train`
- `X_val`, `y_val`
- `X_test`, `y_test`
- `classes`

Split used by the script:

- 70% training
- 15% validation
- 15% testing

Important note:
This script also contains a separate `PRODUCT_CLASSES` list for broader huntable-animal experimentation, but the saved HDF5 training artifact currently reflects the four-class dataset above.

### 3. `src/train_models.py`

Purpose:
Loads `wildlife_data.h5`, fine-tunes three transfer-learning models, evaluates them on the test set, compares results, and writes the main training artifacts.

Models trained:

- `MobileNetV2`
- `ResNet50`
- `EfficientNetB0`

Training strategy:

- Stage 1: train a custom classification head while the base model is frozen.
- Stage 2: unfreeze the top layers and fine-tune them at a lower learning rate.

Run it with:

```bash
python src/train_models.py
```

Main outputs written to `artifacts/`:

- `MobileNetV2_best.keras`
- `MobileNetV2_fine_tuned.keras`
- `ResNet50_best.keras`
- `ResNet50_fine_tuned.keras`
- `EfficientNetB0_best.keras`
- `EfficientNetB0_fine_tuned.keras`
- `fine_tuning_history.png`
- `fine_tuning_comparison.csv`
- `fine_tuning_comparison.png`
- `{best_model}_fine_tuned_confusion.png`
- `best_fine_tuned_model.json`

### 4. `src/SHAP_analysis.py`

Purpose:
Loads the preprocessed HDF5 dataset and the saved best model weights, then generates SHAP-based attribution plots for model interpretation.

Run it with:

```bash
python src/SHAP_analysis.py
```

Expected inputs:

- `artifacts/wildlife_data.h5`
- `artifacts/MobileNetV2_best.keras`
- `artifacts/ResNet50_best.keras`
- `artifacts/EfficientNetB0_best.keras`

Main outputs written to `artifacts/`:

- `shap_MobileNetV2_sample*.png`
- `shap_ResNet50_sample*.png`
- `shap_EfficientNetB0_sample*.png`
- `shap_*_correct*.png`
- `shap_*_incorrect*.png`
- `shap_*_mean.png`
- `shap_model_comparison.png`

### 5. `src/generate_map_images.py`

Purpose:
Builds and exports the static PNG map assets used by the web app instead of generating Plotly maps at request time.

Run it with:

```bash
python src/generate_map_images.py
```

What it generates:

- species range maps
- animal location maps
- district detail maps

Output location:

- `app/assets/generated-maps/`

Important note:
This script depends on Plotly static image export. If `kaleido` is missing, image export will fail.

### 6. `src/model_runtime.py`

Purpose:
Provides the lightweight inference runtime used by the deployed app and exports the runtime assets required by that app.

When imported by the web app, it:

- loads the MobileNetV2 TFLite model through LiteRT
- decodes top ImageNet predictions
- maps matching labels into the app's sold-animal catalog
- falls back to filename-assisted logic when a sold class is not resolved cleanly

When run as a script, it exports:

- `artifacts/mobilenet_v2_imagenet.tflite`
- `artifacts/imagenet_class_index.json`

Run the export with:

```bash
python src/model_runtime.py
```

Important note:
TensorFlow and Keras are required for the export step, but the deployed Render runtime itself uses LiteRT instead of full TensorFlow.

### 7. `src/web_app.py`

Purpose:
Runs the Dash application and connects the UI to the image-classification runtime and the pre-generated map assets.

Main web app features:

- animal sale catalog pages
- animal detail pages
- district detail pages
- South African range-map context
- image upload classification
- sold versus not-sold result cards

Supported upload types:

- JPG
- PNG
- WEBP

Run it locally with:

```bash
python src/web_app.py
```

Local default:

- host: `0.0.0.0`
- port: `8050`

Production entrypoint:

```bash
gunicorn src.web_app:server --workers 1 --threads 2 --timeout 180
```

## Sold Catalog Used By The Web App

The deployed app currently treats these ten species as the configured sold catalog:

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

Anything outside that configured catalog is routed to a not-sold result.

## Notebooks

The notebooks support exploration, analysis, reporting, and map-preview workflows. They are useful for understanding the project, but they are not required to run the deployed web app.

### `notebooks/EDA.ipynb`

Exploratory analysis notebook for the image dataset. It sets up class metadata, parses image filenames, and inspects dataset-level characteristics such as class structure and sample content.

### `notebooks/Comparison.ipynb`

Evaluation-oriented notebook that loads saved models and runs direct comparison steps on them. It appears to capture raw model evaluation output and is useful for quick testing or side-by-side checking during experimentation.

### `notebooks/Model Comparisons.ipynb`

Artifact-driven reporting notebook. It reads saved outputs from `artifacts/`, presents the best-model summary, and acts as a cleaner reporting surface for the final comparison results.

### `notebooks/SHAP_analysis.ipynb`

Notebook version of the SHAP workflow. It loads the saved HDF5 dataset and existing artifacts, then explores prediction explanations interactively.

### `notebooks/Generate Map Images.ipynb`

Interactive companion to `src/generate_map_images.py`. It helps inspect output locations, run the map-generation process, and preview generated PNG assets.

## Render Deployment

This project includes `render.yaml` for deployment.

Configured Render settings:

- runtime: `python`
- plan: `free`
- build command: `pip install -r requirements-render.txt`
- start command: `gunicorn src.web_app:server --workers 1 --threads 2 --timeout 180`
- environment variable: `PYTHON_VERSION=3.12.10`

Deployment notes:

- The hosted site expects `artifacts/mobilenet_v2_imagenet.tflite` and `artifacts/imagenet_class_index.json` to exist.
- The hosted site also expects the generated map assets to be present under `app/assets/generated-maps/`.
- Render is intended for inference and UI hosting, not for model training.

## Important Project Notes

- The research training pipeline and the live website runtime are not the same thing.
- The saved fine-tuned training pipeline currently works on a four-class dataset: buffalo, elephant, rhino, and zebra.
- The live website uses a generic MobileNetV2 ImageNet TFLite model plus label mapping and fallback logic to support a ten-species sold catalog.
- If the LiteRT runtime cannot be loaded, the app falls back to deterministic filename matching.
- Static map images are generated ahead of time so the deployed app does not have to build Plotly figures during a request.

## Recommended Usage Order

If you want the full project workflow, use this order:

1. Prepare or inspect image preprocessing helpers in `src/prepare_data.py`.
2. Build the dataset artifact with `python src/pre_process_data.py`.
3. Train and compare the models with `python src/train_models.py`.
4. Explain model predictions with `python src/SHAP_analysis.py`.
5. Generate map assets with `python src/generate_map_images.py`.
6. Export the runtime assets with `python src/model_runtime.py`.
7. Launch the site with `python src/web_app.py`.

## Summary

This repository contains the full workflow for dataset preparation, model training, explainability, runtime asset export, and a Dash deployment surface. For training and notebooks, use the broader experimentation environment. For deployment, use the LiteRT-based runtime assets, pre-generated map images, and the Dash entrypoint defined in `src/web_app.py`.
