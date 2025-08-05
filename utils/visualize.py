import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== Visualization ==================
def plot_components(method, components, original_image_shape, n_components=25, save_path=None):
    n_cols = 5
    n_rows = (n_components + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))

    # Determine the expected flattened size based on the original image shape
    expected_flat_size_color = original_image_shape[0] * original_image_shape[1] * original_image_shape[2] if len(original_image_shape) == 3 else 0
    expected_flat_size_gray = original_image_shape[0] * original_image_shape[1] if len(original_image_shape) == 2 else 0

    for i in range(n_rows * n_cols):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        if i < components.shape[0]:
            comp_img_flat = components[i]

            if comp_img_flat.shape[0] == expected_flat_size_color and len(original_image_shape) == 3:
                 comp_img_color = comp_img_flat.reshape(original_image_shape)
                 comp_img = cv2.cvtColor(comp_img_color.astype(np.float32), cv2.COLOR_RGB2GRAY)
            elif comp_img_flat.shape[0] == expected_flat_size_gray and len(original_image_shape) == 2:
                 comp_img = comp_img_flat.reshape(original_image_shape)
            else:
                 # Handle unexpected size (e.g., if components are not image-like)
                 print(f"Warning: Component size {comp_img_flat.shape[0]} does not match expected flattened image sizes.")
                 ax.axis('off')
                 continue

            ax.imshow(comp_img, cmap='gray')
            ax.set_title(f"{method} {i+1}")
        else:
            ax.axis('off')

        ax.axis('off')

    plt.suptitle(f"{method} Components (Top {n_components})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 1])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_fisherfaces(pca, lda, original_image_shape, n_components=25, save_path=None):
    fisherfaces = np.dot(lda.scalings_.T, pca.components_)
    fisherfaces = fisherfaces[:n_components]
    plot_components("Fisherface", fisherfaces, original_image_shape, n_components, save_path)

def plot_metrics(result, save_path=None):
    # Isi DataFrame dengan hasil
    rows = []
    for metrics in result:
        rows.append({
            'Model': metrics['classifier'] + ' x ' + metrics['model'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1']
        })

    summary_df = pd.DataFrame(rows)

    # Set figure size and style

    plt.figure(figsize=(10, 6))
    summary_df.set_index('Model').plot(kind='bar', figsize=(10, 6), colormap='tab10')

    # Title and labels
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)  # Score ranges from 0 to 1
    plt.grid(axis='y')
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()