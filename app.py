import uvicorn
import uuid
import aiofiles
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.utils import to_categorical

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathlib import Path

from dto.train import Train
from model.classifier import train_and_evaluate_models
from model.cnn import create_cnn, plot_cnn_feature_maps, plot_cnn_matrics
from model.feature_extraction import extract_pca, extract_lda, extract_ica
from utils.data_load import is_allowed_file, load_images_from_folder
from utils.visualize import plot_components, plot_metrics

app = FastAPI()

# ================== Configuration ==================
IMAGE_SIZE = (244, 244)  # Resize all images to this size
N_COMPONENTS_PCA = 50
N_COMPONENTS_LDA = 3
N_COMPONENTS_ICA = 50
COLOR_MODE = "rgb"  # Change to "gray" for grayscale; "rgb" for color
UPLOAD_DIR = Path("storage/dataset")
OUTPUT_DIR = Path("storage/output")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB per file
MAX_TOTAL_SIZE = 200 * 1024 * 1024  # 200 MB total for all files
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# Create upload directory if not exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CORS (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the directory to serve static files
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

# ================== Main Execution ==================
@app.post("/upload-images")
async def upload_images(
        category: str = Form(...),
        label: str = Form(...),
        files: list[UploadFile] = File(...)
):
    if not files or all(f.filename == "" for f in files):
        return JSONResponse(content={"success": False, "message": "No files uploaded"}, status_code=400)

    total_size = 0
    saved_files = []

    for file in files:
        # Validate file extension
        if not is_allowed_file(file.filename, ALLOWED_EXTENSIONS):
            return JSONResponse(content={"success": False, "message": f"File type not allowed: {file.filename}"}, status_code=400)

        # Read file to check size
        contents = await file.read()
        total_size += len(contents)

        if len(contents) > MAX_FILE_SIZE:
            return JSONResponse(content={"success": False, "message": f"File {file.filename} exceeds the maximum size of {MAX_FILE_SIZE / (1024*1024):.1f} MB"}, status_code=413)

        # Reset file pointer after reading
        await file.seek(0)

    if total_size > MAX_TOTAL_SIZE:
        return JSONResponse(content={"success": False, "message": f"Total upload size exceeds the limit of {MAX_TOTAL_SIZE / (1024*1024):.1f} MB"}, status_code=413)

    # Save files
    # path = category + "_" + label
    upload_path = UPLOAD_DIR / category / label.lower()
    upload_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    for file in files:
        try:
            # Extract extension
            ext = Path(file.filename).suffix
            unique_filename = f"{Path(file.filename).stem}_{uuid.uuid4().hex}{ext}"
            file_path = upload_path / unique_filename

            async with aiofiles.open(file_path, "wb") as f:
                await f.write(await file.read())
            saved_files.append(str(file_path))
        except Exception as e:
            return JSONResponse(content={"success": False,
                                     "message": f"Error saving file {file.filename}: {str(e)}"}, status_code=500)
        finally:
            await file.close()

    return JSONResponse(
        content={
            "success": True,
            "message": f"Successfully uploaded {len(saved_files)} files",
            "data" : {
                "files": saved_files,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
        },
        status_code=201
    )

@app.post("/train")
async def train(request: Train):
    try:
        print("[INFO] Loading dataset...")
        dataset_path = request.path
        if request.path is None:
            dataset_path = "storage/dataset/"+request.category

        X, y = load_images_from_folder(dataset_path, image_size=IMAGE_SIZE, color_mode=COLOR_MODE)
        print(f"[INFO] Loaded {len(X)} images with shape {X.shape[1:]}")

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=1 - request.trainDataPercentage, random_state=42, stratify=y_encoded
        )

        if (request.featureMethod == "CNN"):
            # Normalize images for CNN
            X_train_norm = X_train.astype('float32') / 255.0
            X_test_norm = X_test.astype('float32') / 255.0
        else:
            # Flatten data for classical ML models
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)

        # ================== Feature Extraction ==================
        results = []
        plot_path = str(OUTPUT_DIR) + '\\plot_component.png'
        graph_path = str(OUTPUT_DIR) + '\\confusion_matric_graph.png'

        match request.featureMethod:
            case "PCA":
                # 2. PCA
                print("[INFO] Applying PCA...")
                print("Number of samples (n_samples):", X_train.shape[0])
                print("Number of features (n_features):", X_train.shape[1])
                print("Max n_components allowed:", min(X_train.shape[0], X_train.shape[1]))
                n_components = min(N_COMPONENTS_PCA, X_train.shape[0], X_train.shape[1])
                X_train_pca, pca = extract_pca(X_train, n_components)
                X_test_pca = pca.transform(X_test_flat)
                plot_components("Eigenface", pca.components_, IMAGE_SIZE + (3,) if COLOR_MODE == "rgb" else IMAGE_SIZE,
                                save_path=plot_path)
                results = train_and_evaluate_models(X_train_pca, y_train, X_test_pca, y_test, "PCA", request.classifierMethods)
                plot_metrics(results, graph_path)
            case "LDA":
                # 3. LDA
                print("[INFO] Applying LDA...")
                X_train_lda, lda = extract_lda(X_train, y_train, len(np.unique(y_train)) - 1)
                X_test_lda = lda.transform(X_test_flat)
                plot_components("LDA Component", lda.scalings_.T,
                                IMAGE_SIZE + (3,) if COLOR_MODE == "rgb" else IMAGE_SIZE,
                                save_path=plot_path)

                results = train_and_evaluate_models(X_train_lda, y_train, X_test_lda, y_test, "LDA", request.classifierMethods)
                plot_metrics(results, graph_path)
            case "ICA":
                # 5. ICA
                print("[INFO] Applying ICA...")
                print("Number of samples (n_samples):", X_train.shape[0])
                print("Number of features (n_features):", X_train.shape[1])
                print("Max n_components allowed:", min(X_train.shape[0], X_train.shape[1]))
                n_components = min(N_COMPONENTS_ICA, X_train.shape[0], X_train.shape[1])
                X_train_ica, ica = extract_ica(X_train, n_components)
                X_test_ica = ica.transform(X_test_flat)
                plot_components("ICA Component", ica.components_,
                                IMAGE_SIZE + (3,) if COLOR_MODE == "rgb" else IMAGE_SIZE,
                                save_path=plot_path)

                results = train_and_evaluate_models(X_train_ica, y_train, X_test_ica, y_test, "ICA", request.classifierMethods)
                plot_metrics(results, graph_path)
            case "CNN":
                # ================== CNN Training ==================
                print("\n[INFO] Training CNN on RGB images...")
                num_classes = len(np.unique(y_train))
                cnn_input_shape = IMAGE_SIZE + (3,) if COLOR_MODE == "rgb" else IMAGE_SIZE + (1,)
                cnn_model = create_cnn(input_shape=cnn_input_shape, num_classes=num_classes)

                y_train_cat = to_categorical(y_train)
                y_test_cat = to_categorical(y_test)

                history = cnn_model.fit(
                    X_train_norm, y_train_cat,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=1
                )

                test_loss, test_acc = cnn_model.evaluate(X_test_norm, y_test_cat, verbose=0)
                print(f"\nCNN Test Accuracy: {test_acc:.4f}")

                y_pred_cnn = np.argmax(cnn_model.predict(X_test_norm), axis=1)
                print("\nClassification Report (CNN):")
                results_dic = classification_report(y_test, y_pred_cnn, target_names=le.classes_, output_dict=True, zero_division=0)
                overall_accuracy = results_dic['accuracy']  # Get the overall accuracy

                for label, metrics in results_dic.items():
                    # Only process metrics for individual classes (which are dictionaries)
                    if isinstance(metrics, dict):
                        results.append({
                            'classifier': str(label),  # Convert numpy string to regular string
                            'model': 'CNN',
                            'accuracy': overall_accuracy,  # Include overall accuracy for each class entry
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1': metrics['f1-score']
                        })

                print(results)

                # Visualize CNN feature maps
                idx = np.random.randint(0, len(X_test_norm))
                plot_cnn_feature_maps(cnn_model, X_test_norm[idx], layer_name='Conv1',
                                      save_path=plot_path)
                plot_cnn_matrics(results_dic, le.classes_, graph_path)
            case _:
                # 1. Raw Pixel Features
                print("[INFO] Training on raw pixel features...")
                results = train_and_evaluate_models(X_train_flat, y_train, X_test_flat, y_test, "Raw")
                plot_metrics(results, plot_path)

        data = {
            'plotPath': plot_path,
            'graphPath': graph_path,
            'results': results,
        }

        return JSONResponse(content={"success": True, "message": "Training completed", "data": data})
    except Exception as e:
        return JSONResponse(content={"success": False, "message": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)