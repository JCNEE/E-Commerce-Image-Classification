# E-Commerce-Image-Classification
Image Classification Program for an E-Commerce Dash Website to distribute images of items into categories and identify if a category exists for the images.

## Dependency Files

- `requirements.txt`: local app dependencies for ML-backed image identification.
- `requirements-ml.txt`: extra packages used by training, preprocessing, and SHAP analysis scripts.
- `requirements-render.txt`: optional lightweight Render fallback without the heavy ML runtime.

## Make Image Identification Work

If you want the uploaded image itself to drive the result instead of the deterministic filename fallback:

1. Install the default local dependencies with `pip install -r requirements.txt`.
2. Set `USE_TRAINED_MODEL=true` before starting the app.
3. If you have a real fine-tuned project model for the ten sold animals, set `MODEL_BACKEND=project_model`.
4. Make sure `artifacts/best_fine_tuned_model.json` and the matching `.h5` model file describe the same ten sold animals configured in the catalog.

Important:

- The current checked-in artifact metadata still only covers four classes, so it is not yet a true ten-animal sold-vs-not-sold model.
- `mobilenet_v2` or `resnet50` can be used as generic ImageNet backends for testing, but they are not a replacement for a properly fine-tuned project model.

## Pre-generate Static Map Images

The app now supports pre-generated map images and will automatically use them when they exist in the assets folder. This avoids runtime Plotly map construction on the animal, district, and result pages.

Use this flow locally:

1. Install the extra tooling dependencies with `pip install -r requirements-ml.txt`.
2. Run `python src/generate_map_images.py`.
3. Commit the generated images under `app/assets/generated-maps` if you want Render to serve them directly.

What gets generated:

- Province range maps for each supported species.
- Animal-page maps for each sold animal.
- District-page maps for each configured farm location.

If the generated image for a page is missing, the site now shows an unavailable message instead of building a live Plotly map.

## Render Free Deployment

You do not need to upload `render.yaml` to deploy this project on Render Free. The `render.yaml` file is only a blueprint shortcut. If Render is asking you to pay for blueprint uploads, create the web service manually in the dashboard instead.

Use these settings when creating a new Render Web Service:

- Runtime: `Python`
- Plan: `Free`
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn src.web_app:server --workers 1 --threads 2 --timeout 180`

Add these environment variables:

- `PYTHON_VERSION=3.12.10`
- `USE_TRAINED_MODEL=true`
- `MODEL_BACKEND=mobilenet_v2`

Important:

- This Render setup uses MobileNetV2 for image-driven predictions together with pre-generated static map images.
- The first model-backed prediction can still be slow because Render may need to warm the TensorFlow runtime and load the MobileNet weights.
- If the Free instance still struggles, fall back to `requirements-render.txt` with `USE_TRAINED_MODEL=false` for a lighter non-ML deployment.
