import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import json
import time
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# CONFIGURATION

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Training parameters
BATCH_SIZE = 32
EPOCHS = 30
INITIAL_LEARNING_RATE = 0.0001  

print(f"TensorFlow version: {tf.__version__}")

# LOAD FULL DATASET

def load_full_dataset():
   
    data_path = os.path.join(DATA_DIR, "wildlife_data.h5")
    
    if not os.path.exists(data_path):
        alt_path = os.path.join(ARTIFACTS_DIR, "wildlife_data.h5")
        if os.path.exists(alt_path):
            data_path = alt_path
    
    if not os.path.exists(data_path):
        print(f"No data found at {data_path}")
        return None, None, None, None, None, None, None
    
    with h5py.File(data_path, "r") as f:
        X_train = f["X_train"][:]
        y_train = f["y_train"][:]
        X_val = f["X_val"][:]
        y_val = f["y_val"][:]
        X_test = f["X_test"][:]
        y_test = f["y_test"][:]
        classes = [c.decode("utf-8") for c in f["classes"][:]]
    
    print(f"\n Dataset loaded:")
    print(f"   Training: {len(X_train)} images")
    print(f"   Validation: {len(X_val)} images")
    print(f"   Test: {len(X_test)} images")
    print(f"   Classes: {classes}")
    print(f"   Image shape: {X_train[0].shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, classes


# BUILD FINE-TUNING MODEL

def build_finetune_model(base_model_name, num_classes, input_shape=(224, 224, 3)):
    
    print(f"\n Building {base_model_name} for fine-tuning...")
    
    # Load base model without top layers
    if base_model_name == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown model: {base_model_name}")
    
    
    base_model.trainable = False
    
    # Add custom classification head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"    Model built - Trainable params: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    return model, base_model

# FINE-TUNE MODEL


def fine_tune_model(model, base_model, X_train, y_train, X_val, y_val, model_name):
  
    print(f"\n{'='*60}")
    print(f"FINE-TUNING {model_name}")
    print(f"{'='*60}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(ARTIFACTS_DIR, f'{model_name}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Stage 1: Train new head
    print("\n Stage 1: Training classification head (base model frozen)...")
    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Stage 2: Unfreeze top layers for fine-tuning
    print("\n Stage 2: Fine-tuning top layers")
    
    # Unfreeze the last 30 layers of the base model
    base_model.trainable = True
    
    # Freeze early layers, only train later layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    history2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    full_history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss']
    }
    
    return model, full_history

# EVALUATE MODEL

def evaluate_model(model, model_name, X_test, y_test, class_names):
    print(f"\n Evaluating {model_name} on test set...")
    
    y_true = np.argmax(y_test, axis=1)
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"   Test Accuracy: {accuracy:.2%}")
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'true_labels': y_true,
        'probabilities': y_pred_proba
    }

# PLOT TRAINING HISTORY
def plot_training_history(histories, model_names):
   
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for model_name, history in histories.items():
        axes[0].plot(history['accuracy'], label=f'{model_name} (train)', linewidth=1.5)
        axes[0].plot(history['val_accuracy'], '--', label=f'{model_name} (val)', linewidth=1.5)
    
    axes[0].set_title('Model Accuracy During Fine-tuning', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    for model_name, history in histories.items():
        axes[1].plot(history['loss'], label=f'{model_name} (train)', linewidth=1.5)
        axes[1].plot(history['val_loss'], '--', label=f'{model_name} (val)', linewidth=1.5)
    
    axes[1].set_title('Model Loss During Fine-tuning', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'fine_tuning_history.png'), dpi=150)
    plt.show()

# COMPARE RESULTS
def compare_results(results, class_names):
    """Compare all fine-tuned models."""
    
    # Create comparison table
    df = pd.DataFrame([{
        'Model': name,
        'Test Accuracy': f"{data['accuracy']:.2%}",
        'Accuracy_Value': data['accuracy']
    } for name, data in results.items()])
    df = df.sort_values('Accuracy_Value', ascending=False)
    
    print("\n" + "=" * 60)
    print("FINE-TUNING RESULTS")
    print("=" * 60)
    print(df[['Model', 'Test Accuracy']].to_string(index=False))
    
    # Save to CSV
    df.to_csv(os.path.join(ARTIFACTS_DIR, 'fine_tuning_comparison.csv'), index=False)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    accs = [results[n]['accuracy'] for n in names]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = plt.bar(names, accs, color=colors[:len(names)])
    plt.ylabel('Test Accuracy')
    plt.title('Fine-tuned Model Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'fine_tuning_comparison.png'), dpi=150)
    plt.show()
    
    # Confusion matrix for best model
    best_model = df.iloc[0]['Model']
    best_results = results[best_model]
    
    cm = confusion_matrix(best_results['true_labels'], best_results['predictions'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{best_model} - Confusion Matrix\nAccuracy: {best_results["accuracy"]:.2%}',
              fontsize=12, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, f'{best_model}_fine_tuned_confusion.png'), dpi=150)
    plt.show()
    
    return best_model

# MAIN
def main():
    print("=" * 60)
    

    # Load full dataset
    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_full_dataset()
    
    if X_train is None:
        return
    
    num_classes = len(classes)
    models_to_train = ['MobileNetV2', 'ResNet50', 'EfficientNetB0']
    
    results = {}
    histories = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Processing {model_name}")
        print(f"{'='*60}")
        
        # Build model
        model, base_model = build_finetune_model(model_name, num_classes)
        
        # Fine-tune
        trained_model, history = fine_tune_model(
            model, base_model, X_train, y_train, X_val, y_val, model_name
        )
        
        histories[model_name] = history
        
        # Evaluate
        eval_result = evaluate_model(trained_model, model_name, X_test, y_test, classes)
        results[model_name] = eval_result
        
        # Save final model
        trained_model.save(os.path.join(ARTIFACTS_DIR, f'{model_name}_fine_tuned.h5'))
        print(f" Saved: {ARTIFACTS_DIR}/{model_name}_fine_tuned.h5")
    
    # Plot training histories
    plot_training_history(histories, models_to_train)
    
    # Compare results
    best_model = compare_results(results, classes)
    
    # Print detailed report for best model
    print(f"\n Detailed Classification Report for {best_model}:")
    best_result = results[best_model]
    print(classification_report(
        best_result['true_labels'], 
        best_result['predictions'],
        target_names=classes, 
        digits=3
    ))
    
    # Save best model info
    with open(os.path.join(ARTIFACTS_DIR, 'best_fine_tuned_model.json'), 'w') as f:
        json.dump({
            'best_model': best_model,
            'accuracy': float(results[best_model]['accuracy']),
            'classes': classes,
            'num_classes': num_classes,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test)
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print(" FINE-TUNING COMPLETE!")
    print(f"   Best model: {best_model}")
    print(f"   Test accuracy: {results[best_model]['accuracy']:.2%}")
    print(f"   Models saved to: {ARTIFACTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()