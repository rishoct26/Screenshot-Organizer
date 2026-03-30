# 📱 Screenshot Organizer — Multimodal Classifier

A multimodal deep learning system that classifies mobile screenshots into four categories — **Social**, **Finance**, **Education**, and **Productivity** — by fusing visual features (CLIP) with OCR-extracted text features (multilingual BERT) through a learned gating mechanism.

---

## 🧠 Model Architecture

The final model is a **Gated Fusion** architecture combining two pretrained encoders:

- **CLIP ViT-B/32** — extracts visual features from the screenshot image
- **Multilingual BERT** (`bert-base-multilingual-cased`) — encodes OCR-extracted text
- **Learned Gate** — dynamically weighs the contribution of each modality per sample based on OCR quality

Training is done in two stages:
1. **Stage 1** — Freeze both encoders, train only the gate and classifier head
2. **Stage 2** — Unfreeze the last few layers of each encoder for end-to-end fine-tuning

---

## 📊 Ablation Study

| Model | Val Accuracy |
|---|---|
| TF-IDF + Logistic Regression | baseline |
| ResNet-50 (fine-tuned) | — |
| BERT only (frozen) | — |
| CLIP only (frozen) | — |
| CLIP + BERT Concat Fusion | — |
| **Gated Fusion (final)** | **best** |

---

## 🗂️ Dataset

Images are sourced from three Kaggle datasets and mapped to four target classes:

| Source | Classes Used |
|---|---|
| [uzairkhan45 — Categorized Android App Images](https://www.kaggle.com/datasets/uzairkhan45/categorized-android-apps-images) | social, finance, education, productivity |
| [RVL-CDIP (test split)](https://www.kaggle.com/datasets/pdavpoojan/the-rvlcdip-dataset-test) | finance, education, productivity |
| [Instagram Page Screenshots](https://www.kaggle.com/datasets/bahramjannesarr/instagram-page-screen-shots-in-5-category) | social |
| [Labeled Meme Images](https://www.kaggle.com/datasets/hammadjavaid/6992-labeled-meme-images-dataset) | social |

**Dataset summary:**
- ~600 images per class (2,400 total)
- 70 / 15 / 15 train / val / test split
- Perceptual hash deduplication applied
- OCR extracted via `pytesseract` with quality gating (`high` / `low` / `none`)

---

## 🚀 Getting Started

### 1. Open in Google Colab

This project is designed to run on **Google Colab with a T4 GPU**.

### 2. Install dependencies

```bash
pip install open_clip_torch
pip install transformers==4.41.0
pip install pytesseract==0.3.10
pip install imagehash==4.3.1

apt-get install tesseract-ocr
apt-get install tesseract-ocr-all
```

### 3. Mount Google Drive

All filtered images, OCR data, checkpoints, and metadata are stored persistently in:

```
/content/drive/MyDrive/screenshot_organizer_v2/
├── filtered/          # 600 images per class
├── final/             # train / val / test splits
├── checkpoints/       # saved model weights (.pt)
├── ocr_texts.json     # pre-extracted OCR + quality flags
├── metadata.csv       # image source tracking
└── ablation_results.json
```

### 4. Run the notebook cells in order

| Cell | Description |
|---|---|
| Cell 1 | Installs |
| Cell 2 | Imports, seeds, Drive mount |
| Cell 3 | Create folder structure |
| Cells 4–10 | Dataset download, filtering, dedup, OCR, split |
| Cell 11 (Recovery) | Restore state after runtime reset |
| Cell 12 | Model definitions (ResNet, BERT, CLIP, Concat, Gated) |
| Cell 13 | Shared training loop |
| Cell 14 | Baseline: TF-IDF + Logistic Regression |
| Cell 15 | Baseline: ResNet-50 |
| Cells 16–19 | Ablation: CLIP-only, BERT-only, Concat, Gated Fusion |
| Cell 20 | Full test evaluation + confusion matrix |
| Cell 21 | Visual error analysis |
| Cell 22 | Ablation results table + training curves |
| Cell 23 | Interactive classification GUI (ipywidgets) |

---

## 🖥️ Interactive Demo (GUI)

The final cell launches an **ipywidgets-based GUI** directly inside Colab:

1. Upload a screenshot (`.png`, `.jpg`, `.jpeg`)
2. Click **🔍 Classify**
3. See the predicted class, confidence score, and probability bar chart
4. OCR text extracted from the image is also displayed

---

## 📁 Project Structure

```
screenshot_organizer_v2/
├── filtered/
│   ├── social/
│   ├── finance/
│   ├── education/
│   └── productivity/
├── final/
│   ├── train/
│   ├── val/
│   └── test/
├── checkpoints/
│   ├── gated-fusion_best.pt
│   ├── clip-only_best.pt
│   └── ...
├── ocr_texts.json
├── metadata.csv
├── ablation_results.json
├── ablation_results.png
├── clip_only_confusion_matrix.png
└── clip_only_calibration.png
```

---

## 🛠️ Tech Stack

- **PyTorch** — model training and inference
- **OpenCLIP** (`ViT-B/32`, OpenAI weights) — visual encoder
- **HuggingFace Transformers** — multilingual BERT tokenizer and model
- **pytesseract** — OCR text extraction
- **scikit-learn** — TF-IDF baseline, metrics
- **Pillow / OpenCV** — image processing and `.tif` conversion
- **imagehash** — perceptual hash deduplication
- **ipywidgets** — interactive Colab GUI
- **matplotlib / seaborn** — evaluation plots

---

## 📄 License

MIT
