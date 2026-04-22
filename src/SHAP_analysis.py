"""
SHAP_analysis.py
----------------
Generates pixel-attribution heatmaps for the three fine-tuned CNN models
produced by train_models.py, using SHAP's GradientExplainer.

Now loads train/val/test splits directly from wildlife_data.h5
instead of expecting image folders.
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
import h5py

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(BASE_DIR)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

h5_path = os.path.join(ARTIFACTS_DIR, "wildlife_data.h5")
if not os.path.exists(h5_path):
    raise FileNotFoundError(f"{h5_path} not found. Please run pre_process_data.py first.")

try:
    with h5py.File(h5_path, "r") as f:
        print("HDF5 file keys:", list(f.keys()))
except Exception as e:
    raise RuntimeError(f"{h5_path} is invalid or corrupted. "
                       f"Please delete it and rerun pre_process_data.py.\nError: {e}")


def _artifact(filename):
    return os.path.join(ARTIFACTS_DIR, filename)

# Maps friendly name → saved .h5 path from train_models.py
MODELS = {
    "MobileNetV2"   : _artifact("MobileNetV2_best.h5"),
    "ResNet50"      : _artifact("ResNet50_best.h5"),
    "EfficientNetB0": _artifact("EfficientNetB0_best.h5"),
}

SHAP_OUTPUT_TEMPLATE = _artifact("shap_{model_name}.png")

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE     = (224, 224)
BATCH        = 32
N_BACKGROUND = 50   # background images for SHAP baseline
N_SAMPLES    = 3     # images for Section A heatmaps
N_CORRECT    = 3     # correct examples for Section B
N_INCORRECT  = 3     # incorrect examples for Section B
N_MEAN       = 20    # images for Section C mean map


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def run_shap_cnn(model, background, images):
    explainer   = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(images)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return np.array(shap_values)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Standard pixel heatmaps (improved overlay)
# ══════════════════════════════════════════════════════════════════════════════

def section_a_heatmaps(model_name, model, background, X_test):
    print(f"  [A] Computing heatmaps ({N_SAMPLES} images)...")
    sample_imgs = X_test[:N_SAMPLES]
    shap_vals   = run_shap_cnn(model, background, sample_imgs)

    # Normalize SHAP values for visibility
    shap_vals = shap_vals / np.max(np.abs(shap_vals))

    # Plot each sample with vivid overlay
    for i, img in enumerate(sample_imgs):
        shap_map = np.mean(np.abs(shap_vals[i]), axis=-1)  # absolute SHAP per pixel
        plt.figure(figsize=(4,4))
        plt.imshow(img)  # original image
        plt.imshow(shap_map, cmap="inferno", alpha=0.6)  # vivid overlay
        plt.axis("off")
        plt.title(f"{model_name} — Sample {i+1}")
        out = _artifact(f"shap_{model_name}_sample{i+1}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [A] Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Correct vs incorrect predictions (improved overlay)
# ══════════════════════════════════════════════════════════════════════════════

def section_b_correct_vs_incorrect(model_name, model, background, X_test, y_test, classes):
    print(f"  [B] Running correct vs incorrect analysis...")

    raw_preds = model.predict(X_test, verbose=0)

    # Case 1: sigmoid (binary classification)
    if raw_preds.ndim == 2 and raw_preds.shape[1] == 1:
        preds = (raw_preds.ravel() > 0.5).astype(int)
    # Case 2: softmax (multi-class classification)
    else:
        preds = np.argmax(raw_preds, axis=1)

    correct_idx = np.where(preds == y_test)[0]
    wrong_idx   = np.where(preds != y_test)[0]

    print(f"      Correct  : {len(correct_idx)} / {len(y_test)}")
    print(f"      Incorrect: {len(wrong_idx)} / {len(y_test)}")

    # Show class names for first few examples
    if len(correct_idx):
        for i in correct_idx[:N_CORRECT]:
            print(f"   ✓ Correct: Predicted {classes[preds[i]]}, True {classes[y_test[i]]}")

    if len(wrong_idx):
        for i in wrong_idx[:N_INCORRECT]:
            print(f"   ✗ Incorrect: Predicted {classes[preds[i]]}, True {classes[y_test[i]]}")

    # Plot SHAP for correct examples
    c_imgs = X_test[correct_idx[:N_CORRECT]]
    if len(c_imgs):
        c_shap = run_shap_cnn(model, background, c_imgs)
        c_shap = c_shap / np.max(np.abs(c_shap))  # normalize
        for i, img in enumerate(c_imgs):
            shap_map = np.mean(np.abs(c_shap[i]), axis=-1)
            plt.figure(figsize=(4,4))
            plt.imshow(img)
            plt.imshow(shap_map, cmap="inferno", alpha=0.6)
            plt.axis("off")
            plt.title(f"{model_name} — Correct {classes[y_test[correct_idx[i]]]}")
            out = _artifact(f"shap_{model_name}_correct{i+1}.png")
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  [B] Saved: {out}")

    # Plot SHAP for incorrect examples
    w_imgs = X_test[wrong_idx[:N_INCORRECT]] if len(wrong_idx) else np.array([])
    if len(w_imgs):
        w_shap = run_shap_cnn(model, background, w_imgs)
        w_shap = w_shap / np.max(np.abs(w_shap))  # normalize
        for i, img in enumerate(w_imgs):
            shap_map = np.mean(np.abs(w_shap[i]), axis=-1)
            plt.figure(figsize=(4,4))
            plt.imshow(img)
            plt.imshow(shap_map, cmap="inferno", alpha=0.6)
            plt.axis("off")
            plt.title(f"{model_name} — Incorrect (Pred {classes[preds[wrong_idx[i]]]}, True {classes[y_test[wrong_idx[i]]]})")
            out = _artifact(f"shap_{model_name}_incorrect{i+1}.png")
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  [B] Saved: {out}")
    else:
        print(f"  [B] No incorrect predictions — skipping incorrect plot.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — Mean |SHAP| global importance map
# ══════════════════════════════════════════════════════════════════════════════

def section_c_mean_shap(model_name, model, background, X_test):
    print(f"  [C] Computing mean |SHAP| map ({N_MEAN} images)...")
    imgs_batch = X_test[:N_MEAN]
    shap_vals  = run_shap_cnn(model, background, imgs_batch)

    mean_shap = np.mean(np.abs(shap_vals), axis=(0, 3))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(mean_shap, cmap="hot")
    axes[0].set_title("Mean |SHAP| across test set")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    avg_img = imgs_batch.mean(axis=0)
    axes[1].imshow(avg_img)
    axes[1].imshow(mean_shap, cmap="hot", alpha=0.5)
    axes[1].set_title("Overlay on average image")
    axes[1].axis("off")

    fig.suptitle(f"{model_name} — Global attribution map", fontsize=12)
    plt.tight_layout()
    out = _artifact(f"shap_{model_name}_mean.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [C] Saved: {out}")
    return mean_shap


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — Side-by-side model comparison
# ══════════════════════════════════════════════════════════════════════════════

def section_d_model_comparison(mean_shap_maps):
    available = {k: v for k, v in mean_shap_maps.items() if v is not None}
    if len(available) < 2:
        print("[D] Need at least 2 models to compare — skipping.")
        return

    print(f"[D] Building model comparison plot ({len(available)} models)...")
    n    = len(available)
    vmax = max(v.max() for v in available.values())

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, mean_shap) in zip(axes, available.items()):
        im = ax.imshow(mean_shap, cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(name, fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Mean |SHAP| — Model comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    out = _artifact("shap_model_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[D] Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("SA Ranch — SHAP Analysis")
    print("=" * 60)

    # Load preprocessed HDF5 data
    print("\nLoading preprocessed HDF5 data...")
    with h5py.File(os.path.join(ARTIFACTS_DIR, "wildlife_data.h5"), "r") as f:
        X_train = f["X_train"][:]
        y_train = f["y_train"][:]
        X_val   = f["X_val"][:]
        y_val   = f["y_val"][:]
        X_test  = f["X_test"][:]
        y_test  = f["y_test"][:]
        classes = [c.decode("utf-8") for c in f["classes"][:]]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Build SHAP background set once (shared across all models)
    print(f"\nBuilding background set ({N_BACKGROUND} images)...")
    background = X_train[:N_BACKGROUND]
    print(f"  Background shape: {background.shape}")

    # Use full test set for Sections B and C
    X_test_all = X_test
    y_test_all = np.argmax(y_test, axis=1)  # convert one-hot back to integers
    print(f"  Test shape: {X_test_all.shape} | Labels: {np.bincount(y_test_all)}")

    mean_shap_maps = {}

    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"\n[SKIP] {model_name}: {model_path} not found")
            mean_shap_maps[model_name] = None
            continue

        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        model = tf.keras.models.load_model(model_path)
        print(f"  Input shape: {model.input_shape}")

        section_a_heatmaps(model_name, model, background, X_test_all)
        section_b_correct_vs_incorrect(model_name, model, background, X_test_all, y_test_all, classes)
        mean_shap_maps[model_name] = section_c_mean_shap(model_name, model, background, X_test_all)

        tf.keras.backend.clear_session()

    # Section D — comparison across all models
    print(f"\n{'='*60}")
    section_d_model_comparison(mean_shap_maps)

    # Artifact summary
    print(f"\n{'='*60}")
    print("SHAP artifact check")
    print("-" * 50)
    all_files = sorted(glob.glob(_artifact("shap_*.png")))
    for path in all_files:
        size_kb = os.path.getsize(path) / 1024
        print(f"  ✓  {size_kb:6.1f} KB  {path}")
    print("-" * 50)
    print(f"Total: {len(all_files)} file(s)")
    print("=" * 60)
