
# Importing libaries
import os
import numpy as np
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
import joblib
#===================================================================================================================
# SECTION 1 — CNN SHAP (GradientExplainer)

def explain_cnn(
    model_path: str = "artifacts/MobileNetV2_best.h5",
    train_gen=None,
    test_gen=None,
    n_background: int = 100,
    n_samples: int = 5,
    output_path: str = "artifacts/shap_cnn.png",
    show: bool = False,
) -> None:
    """
    Generate SHAP pixel-attribution heatmaps for a trained CNN model.

    How it works
    ------------
    SHAP's GradientExplainer works by comparing the model's output on a test
    image against its average output on a set of "background" (reference)
    images. For each pixel it asks:
    "How much did this pixel push the prediction toward 'Sold Animal'
    vs. the average?"

      • RED   pixels → pushed the prediction TOWARD  class 1 (Sold Animal)
      • BLUE  pixels → pushed the prediction TOWARD  class 0 (Not Listed)

    The background set should be a representative sample of training images
    so that the baseline is meaningful.

    Parameters
    ----------
    model_path   : Path to the saved .h5 Keras model file.
    train_gen    : Keras ImageDataGenerator flow for the training set.
                   Used to build the background reference dataset.
    test_gen     : Keras ImageDataGenerator flow for the test set.
                   A batch of images is drawn from here for explanation.
    n_background : Number of training images to use as SHAP background.
                   More = more accurate but slower. 100 is a good default.
    n_samples    : Number of test images to explain (max = batch size).
    output_path  : Where to save the SHAP heatmap PNG.
    show         : If True, display the plot inline (useful in notebooks).

    Returns
    -------
    None — saves the heatmap to output_path.
    """

    # Loading the train CNN model

    print(f"[CNN SHAP] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"[CNN SHAP] Model loaded. Input shape: {model.input_shape}")

    # Building the background dataset
    
    # The background is a small set of training images that represents
    # the "typical" input. SHAP compares each test image against this
    # baseline to compute feature attributions.
    print(f"[CNN SHAP] Building background dataset ({n_background} images)...")
    background_batches = []
    collected = 0

    for batch_x, _ in train_gen:
        background_batches.append(batch_x)
        collected += len(batch_x)
        if collected >= n_background:
            break

    background = np.vstack(background_batches)[:n_background]
    print(f"[CNN SHAP] Background shape: {background.shape}")

    # Creating the GradientExplainer
    
    # GradientExplainer uses gradient information flowing through the
    # network to estimate each pixel's contribution to the output.
    # It is much faster than KernelExplainer for image models.
    print("[CNN SHAP] Creating GradientExplainer...")
    explainer = shap.GradientExplainer(model, background)

    # Selecting the test images to explain
    test_gen.reset()  
    sample_batch, _ = next(test_gen)
    sample_imgs = sample_batch[:n_samples] 
    print(f"[CNN SHAP] Explaining {len(sample_imgs)} test images...")

    # computing SHAP values
    # shap_values is a list with one entry per model output.
    # For binary classification with a single sigmoid output, it has
    # shape: (n_samples, height, width, channels)
    shap_values = explainer.shap_values(sample_imgs)
    print(f"[CNN SHAP] SHAP values computed. Shape: {np.array(shap_values).shape}")

    # Plotting and saving the heatmaps
    # shap.image_plot overlays the attribution heatmap on the original
    # image. Red = important for 'Sold Animal', Blue = against it.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    shap.image_plot(shap_values, sample_imgs, show=False)
    plt.suptitle(
        "CNN SHAP — Which pixels drove the animal classification?\n"
        "Red = pushed toward 'Sold Animal' | Blue = pushed toward 'Not Listed'",
        fontsize=11,
        y=1.02,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[CNN SHAP] Heatmap saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


#===================================================================================================================
# SECTION 2 — RANDOM FOREST SHAP (TreeExplainer)

def explain_random_forest(
    model_path: str = "artifacts/RandomForest_HOG.pkl",
    X_test_path: str = "artifacts/X_test_pca.npy",
    n_samples: int = 50,
    n_top_features: int = 20,
    output_path: str = "artifacts/shap_rf.png",
    show: bool = False,
) -> None:
    """
    Generate a SHAP feature-importance bar chart for the Random Forest + HOG model.

    How it works
    ------------
    SHAP's TreeExplainer is specifically designed for tree-based models
    (Random Forest, XGBoost, LightGBM, etc.). It computes exact SHAP
    values very efficiently by traversing the tree structure.

    For this project the features are HOG + PCA components (300 dimensions).
    Each bar in the chart shows the average absolute SHAP value for that
    PCA component — i.e. how much it shifted predictions on average.

    Note: The PCA components are not directly interpretable as pixels, but
    high-importance components represent the HOG gradient patterns that
    most strongly indicate an animal vs. a non-animal.

    Parameters
    ----------
    model_path     : Path to the saved Random Forest .pkl file.
    X_test_path    : Path to the PCA-reduced test features (.npy file).
    n_samples      : Number of test samples to explain.
    n_top_features : How many top features to display in the bar chart.
    output_path    : Where to save the SHAP bar chart PNG.
    show           : If True, display the plot inline (useful in notebooks).

    Returns
    -------
    None — saves the chart to output_path.
    """

    #Loading the trained Random Forest and test features
    print(f"[RF SHAP] Loading model from: {model_path}")
    rf_model = joblib.load(model_path)

    print(f"[RF SHAP] Loading test features from: {X_test_path}")
    X_test = np.load(X_test_path)
    X_sample = X_test[:n_samples]
    print(f"[RF SHAP] Feature matrix shape: {X_sample.shape}")

    #Creating the TreeExplainer
    print("[RF SHAP] Creating TreeExplainer...")
    explainer = shap.TreeExplainer(rf_model)

    #Compute SHAP values
    print(f"[RF SHAP] Computing SHAP values for {n_samples} samples...")
    shap_values = explainer.shap_values(X_sample)

    sold_animal_shap = shap_values[1]
    print(f"[RF SHAP] SHAP values shape (class 1): {sold_animal_shap.shape}")

    # Ploting and saving the summary bar chart
    # summary_plot with plot_type="bar" shows mean |SHAP value| per feature.
    # The longer the bar, the more that HOG/PCA component influenced
    # the model's decision to classify an image as a "Sold Animal".
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    shap.summary_plot(
        sold_animal_shap,
        X_sample,
        plot_type="bar",
        max_display=n_top_features,
        show=False,
    )
    plt.title(
        f"Top {n_top_features} HOG/PCA Features — Random Forest\n"
        "(Mean |SHAP value| for 'Sold Animal' class)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[RF SHAP] Bar chart saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


#===================================================================================================================
# SECTION 3 — BEESWARM PLOT

def explain_rf_beeswarm(
    model_path: str = "artifacts/RandomForest_HOG.pkl",
    X_test_path: str = "artifacts/X_test_pca.npy",
    n_samples: int = 50,
    n_top_features: int = 15,
    output_path: str = "artifacts/shap_rf_beeswarm.png",
    show: bool = False,
) -> None:
    """
    Generate a SHAP beeswarm plot for the Random Forest model.

    The beeswarm plot shows both the magnitude AND the direction of each
    feature's effect:
      • Each dot  = one test sample
      • X-axis    = SHAP value (positive → toward 'Sold Animal')
      • Colour    = feature value (red = high HOG intensity, blue = low)

    This gives more insight than the bar chart alone — you can see
    whether a high HOG gradient value pushes toward or against the
    'Sold Animal' class.

    Parameters
    ----------
    Same as explain_random_forest().
    """

    print(f"[RF Beeswarm] Loading model: {model_path}")
    rf_model = joblib.load(model_path)
    X_test   = np.load(X_test_path)
    X_sample = X_test[:n_samples]

    print("[RF Beeswarm] Creating TreeExplainer...")
    explainer  = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # summary_plot default (no plot_type argument) = beeswarm
    shap.summary_plot(
        shap_values[1],     
        X_sample,
        max_display=n_top_features,
        show=False,
    )
    plt.title(
        f"SHAP Beeswarm — Random Forest (Sold Animal class)\n"
        "Each dot = one sample | Colour = feature magnitude",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[RF Beeswarm] Plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


#===================================================================================================================
# SECTION 4 — Convenience runner (called when script is run directly)


if __name__ == "__main__":
    """
    When run as a standalone script this block runs the RF SHAP analysis
    only (since the CNN explainer needs live data generators from training).

    To run the CNN explainer, import this module from the notebook and
    pass the train_gen / test_gen objects directly.
    """
    print("=" * 60)
    print("SA Ranch — SHAP Analysis Script")
    print("=" * 60)

    print("\n--- Running Random Forest SHAP (bar chart) ---")
    explain_random_forest(show=False)

    print("\n--- Running Random Forest SHAP (beeswarm) ---")
    explain_rf_beeswarm(show=False)

    print("\nDone. Check the artifacts/ folder for output PNGs.")
    print(
        "To run CNN SHAP, import explain_cnn() from the notebook "
        "and pass your train_gen / test_gen objects."
    )