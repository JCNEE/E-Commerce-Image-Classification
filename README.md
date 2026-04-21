# E-Commerce-Image-Classification
Image Classification Program for an E-Commerce Dash Website to distribute images of items into categories and identify if a category exists for the images.

## Render Free Deployment

You do not need to upload `render.yaml` to deploy this project on Render Free. The `render.yaml` file is only a blueprint shortcut. If Render is asking you to pay for blueprint uploads, create the web service manually in the dashboard instead.

Use these settings when creating a new Render Web Service:

- Runtime: `Python`
- Plan: `Free`
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn src.web_app:server`

Add these environment variables:

- `PYTHON_VERSION=3.12.10`
- `USE_TRAINED_MODEL=false`

Important:

- On Render Free, this project is configured to run in lightweight fallback mode.
- Live TensorFlow/Keras image inference is intentionally disabled on Free to avoid heavy builds and startup failures.
- If you later move to a paid instance or deploy locally with ML enabled, install from `requirements-ml.txt` and set `USE_TRAINED_MODEL=true`.
