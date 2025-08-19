import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.applications import MobileNetV2

# ================== CNN Model ==================
# Define synchronized colors for all plots
metric_colors = {
    'Accuracy': '#1f77b4',   # blue
    'Precision': '#ff7f0e',  # orange
    'Recall': '#2ca02c',     # green
    'F1-Score': '#d62728'    # red
}

def create_cnn(input_shape, num_classes):
    # Input shape must be at least (32, 32, 3), but typically 224x224x3 for ImageNet models
    inputs = Input(shape=input_shape)

    # Load pretrained base model (ImageNet weights)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,  # Remove final classification layer
        input_tensor=inputs,  # Use our input
        pooling='avg'  # Global average pooling
    )

    # Freeze the base model (optional: train only the top layer first)
    base_model.trainable = False

    # Add custom head
    x = base_model.output
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def plot_cnn_feature_maps(model, image, layer_name='conv1', save_path=None):
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    feature_maps = feature_extractor.predict(np.expand_dims(image, axis=0))
    n_features = min(feature_maps.shape[-1], 25)
    n_cols, n_rows = 5, (n_features + 4) // 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    for i in range(n_rows * n_cols):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        if i < n_features:
            ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
            ax.set_title(f'{layer_name} - {i+1}')
        ax.axis('off')
    plt.suptitle(f'Feature Maps: {layer_name}')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()

def plot_cnn_matrics(result_dict, class_labels, save_path=None):
    # Extract precision, recall, f1-score per class (excluding accuracy, macro avg, etc.)
    accuracy = [result_dict["accuracy"] * 100 for _ in class_labels]
    precision = [result_dict[label]["precision"] * 100 for label in class_labels]  # convert to %
    recall = [result_dict[label]["recall"] * 100 for label in class_labels]        # convert to %
    f1 = [result_dict[label]["f1-score"] * 100 for label in class_labels]          # convert to %

    # Plot bar chart
    x = np.arange(len(class_labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    rects0 = ax.bar(x - 1.5 * width, accuracy, width, label='Accuracy', color=metric_colors['Accuracy'])
    rects1 = ax.bar(x - 0.5 * width, precision, width, label='Precision', color=metric_colors['Precision'])
    rects2 = ax.bar(x + 0.5 * width, recall, width, label='Recall', color=metric_colors['Recall'])
    rects3 = ax.bar(x + 1.5 * width, f1, width, label='F1-Score', color=metric_colors['F1-Score'])


    ax.set_ylabel('Scores (%)')
    ax.set_title('CNN Classification Report by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.set_ylim(0, 110)  # percentage scale
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar with its value as %"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.0f}%',   # no decimals
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects0)
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()