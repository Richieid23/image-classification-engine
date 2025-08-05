# 🧠 Image Classification Engine

A modular, scalable, and customizable engine for training, evaluating, and deploying image classification models using deep learning. Built with extensibility in mind, this project supports configuration-based pipelines, real-time inference via API, and clean modular components.

---

## 🚀 Features

- ✅ Dataset loading, preprocessing, and augmentation
- ✅ Model training and evaluation with metrics
- ✅ Real-time inference via FastAPI
- ✅ Configuration-based training pipeline
- ✅ Exportable models for deployment
- ✅ Supports multiple backbones and classifier heads

---
## ⚙️ Installation

Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/Richieid23/image-classification-engine.git
cd image-classification-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
---

## 🌐 Running the API Server

Serve the model via a REST API using FastAPI and Uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- `http://localhost:8000/upload-images` – Upload images dataset
- `http://localhost:8000/train` – Train the model

### 🧪 Example cURL Request

```bash
curl --location 'http://localhost:8000/upload-images' \
--form 'category="1"' \
--form 'label="sunflower"' \
--form 'files=@"path/to/your/file"' \
--form 'files=@"path/to/your/file"'
```

Example Response:
```json
{
    "message": "Successfully uploaded 50 files",
    "files": [
        "storage\\dataset\\1_sunflower\\4745991955_6804568ae0_n_10f10461d5324ac69e3570b5d85a7eb1.jpg",
        "storage\\dataset\\1_sunflower\\4746638094_f5336788a0_n_96e2325ff85441e7b0e3d9be81150604.jpg"
    ],
    "total_size_mb": 4.42
}
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙋 Contact

For issues, ideas, or contributions, feel free to:

- Open an issue on GitHub
- Submit a pull request
- Contact me via [GitHub profile](https://github.com/Richieid23)

---