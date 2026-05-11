# 🧠 XAI Emotion Recognition

**Beyond Heatmaps: Grounded Textual Explanations for Facial Emotion Recognition via Multimodal Feature Fusion and Vision-Language Models**

> A research-grade multimodal pipeline that produces **natural language explanations** for facial emotion predictions, grounded in Grad-ECLIP attention evidence and FACS-based geometric feature analysis — making XAI truly autonomous and interpretable.

---

## 🔥 Key Innovation

Traditional XAI methods (SHAP, LIME, Grad-CAM) produce **visual explanations** (heatmaps) that still require human experts to interpret. This system **eliminates the human-in-the-loop** by generating clinical textual explanations that explain *why* a specific emotion was predicted.

**Example output:**
```
Input:  Face Image

Output:
  Emotion:     Happy (87% confidence)

  Explanation: "The face shows happiness primarily because lip corners are
  pulled upward (AU12), cheeks are raised (AU6), and the model's attention
  is focused on the mouth and eye regions — consistent with the FACS
  indicators of a genuine Duchenne smile."

  Visuals:     Grad-ECLIP heatmap  +  facial landmark overlay  +  XAI panel (PNG)
```

---

## 📐 Architecture

```
Input Image
   ↓
MediaPipe FaceLandmarker  (Tasks SDK — face_landmarker.task)
   ↓  468 normalized landmarks (x, y, z)
┌────────────────────────────────────────────────────────────────┐
│  Branch 1: AU Extractor          Branch 2: Vision CNN          │
│  (geometric distances → FACS)    (ConvNeXt / RegNetY)          │
│                                       ↓                        │
│                                       ↓                        │
│                               Emotion Logits (8 classes)       │
│                                       ↓                        │
│                               Grad-ECLIP attention map         │
└────────────────────────────────────────────────────────────────┘
         ↓                                   ↓
   Active AUs + descriptions        Semantic attention regions
         ↓                                   ↓
   ┌─────────────── PromptBuilder ───────────────┐
   │  Emotion + Confidence + AUs + Regions + FACS│
   └─────────────────────────────────────────────┘
                        ↓
         Qwen2.5-0.5B-Instruct  (text-only LLM)
                        ↓
   Output: { emotion, confidence, explanation, heatmap, landmarks, XAI panel }
```

### Component Summary

| Component | Implementation | Notes |
|---|---|---|
| Face detection | MediaPipe Tasks SDK `FaceLandmarker` | `.task` blob required (see Setup) |
| Landmark extraction | 468 normalized (x, y, z) landmarks | Truncated from 478 if irises detected |
| AU extraction | Geometric distances → FACS thresholds | Calibrated on FER2013 |
| **Primary Dataset** | **AffectNet (8 classes)** | Stratified 80/10/10 split |
| Emotion classifier | **ConvNeXt-Tiny** / **RegNetY-800MF** | Primary high-accuracy backbones |
| Baseline classifiers | **ResNet-18**, **EfficientNet-b4** | Standard vision backbones |
| Attention map | **Grad-ECLIP** (custom) or Standard Grad-CAM | Grad-ECLIP is primary |
| Region parser | Heatmap → 11 semantic facial regions | Threshold: top 30% intensity |
| Explanation LLM | **Qwen/Qwen2.5-0.5B-Instruct** | Text-only; no quantization needed |
| Visualization | Combined XAI panel (original + landmarks + heatmap) | Saved as `xai_panel.png` |

---

## 🛠️ Setup

### Prerequisites

- **Python 3.10–3.12** (3.12 recommended)
- NVIDIA GPU with **6 GB+ VRAM** (tested on RTX 4050)
- [`uv`](https://docs.astral.sh/uv/) — fast Python package manager

### 1. Install `uv`

```bash
# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Install

```bash
git clone https://github.com/ad23b1012/XAI.git
cd XAI
uv sync
```

> **What `uv sync` does:** Downloads the correct Python version if missing, creates `.venv`, and installs all dependencies — including PyTorch with CUDA 12.1 — in one step.

### 3. Download the MediaPipe Face Landmarker Model

The face detector requires a pre-built `.task` blob from Google:

```bash
# Create the models directory
mkdir -p src/face_detection/models

# Download the float16 model (recommended)
curl -L https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
     -o src/face_detection/models/face_landmarker.task
```

> This file is **git-ignored** — every developer must download it once.

---

## 📦 Dataset Setup

### AffectNet (Primary Training Dataset)

The project uses a stratified 80/10/10 split on AffectNet-8. The directory structure should look like this:

```
dataset_cropped/
├── anger/
├── contempt/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

### FER2013 (Secondary / Legacy)
The training script also supports FER2013:
```
data/FER/
├── train/
└── test/
```

---

## 🚀 Usage

> **Always prefix commands with `uv run`** when using the `uv` managed environment.

### 1. Train the Emotion Classifier

```bash
# High Accuracy Primary Model (ConvNeXt-Tiny)
uv run python scripts/train_classifier.py \
    --model convnext_tiny \
    --dataset affectnet \
    --data-path dataset_cropped \
    --epochs 50 \
    --lr 1e-4

# Fast Training Model (RegNetY-800MF)
uv run python scripts/train_classifier.py \
    --model regnet_y_800mf \
    --dataset affectnet \
    --data-path dataset_cropped \
    --epochs 50 \
    --lr 1e-3
```

**Key training features:**
- **Class-Weighted Focal Loss** — automatically balances FER2013's minority classes (disgust, fear)
- **AMP (mixed precision)** — enabled by default for RTX GPUs; disable with `--no-amp`
- **Auto-resume** — training resumes from `checkpoints/<model>_last.pth` if it exists
- **Cosine LR schedule** with 5-epoch warmup

Checkpoints are saved to `checkpoints/` after each epoch (`_last.pth` and `_best.pth`).

### 2. Evaluate a Trained Model

```bash
uv run python scripts/evaluate.py \
    --model convnext_tiny \
    --dataset affectnet \
    --data-path dataset_cropped \
    --checkpoint checkpoints/convnext_tiny_best.pth \
    --split test
```

Outputs a per-class classification report and confusion matrix PNG to `outputs/`.

### 3. Extract Facial Landmarks

Pre-extract landmarks from images for faster geometric processing (if needed for multimodal fusion):

```bash
uv run python scripts/extract_landmarks.py --image path/to/face.jpg
```

### 4. Run the Full XAI Pipeline

```bash
# Full pipeline: face detection → emotion → attention → Qwen explanation
uv run python scripts/demo.py --image path/to/face.jpg --model convnext_tiny

# Use a specific baseline
uv run python scripts/demo.py --image path/to/face.jpg --model regnet_y_800mf
```

### 5. Batch Utilities

**Run XAI tests across all 8 classes:**
```bash
uv run python scripts/batch_demo.py --data-path dataset_cropped --num-per-class 1
```

**Evaluate an unlabelled folder of images:**
```bash
uv run python scripts/eval_test_folder.py --data-path Test --output-dir outputs
```

All output files (JSON result, face crop, XAI panel PNG) are saved under `outputs/<model_name>/`.

---

## 🔄 Pipeline Flow (Step-by-Step)

```
Step 1  →  MediaPipe FaceLandmarker detects face + extracts 468 landmarks
Step 2  →  AUExtractor computes geometric distances → activates FACS Action Units
Step 3  →  CNN Classifier (ConvNeXt-Tiny/RegNetY) predicts emotion probabilities
Step 4  →  Grad-ECLIP generates attention heatmap from backward gradients
Step 5  →  RegionParser maps heatmap intensity → semantic facial regions
Step 6  →  [VRAM swap] Classifier unloaded → Qwen2.5-0.5B-Instruct loaded
Step 7  →  PromptBuilder assembles evidence prompt (emotion + AUs + regions + FACS)
Step 8  →  Qwen generates the textual explanation (greedy decode, ≤150 tokens)
Step 9  →  Visualization: landmark overlay + heatmap overlay + combined XAI panel saved
```

---

## 📊 Results

### Emotion Classification Accuracy (AffectNet-8)

| Model | Val Accuracy (Best) | Status |
|---|:---:|---|
| **ConvNeXt-Tiny** | *Training...* | New Primary Model (82.5% ImageNet) |
| **RegNetY-800MF** | *Training...* | Fast Baseline (76.4% ImageNet) |
| ResNet-18 | 71.90% | SOTA Baseline |
| EfficientNet-b4 | 71.58% | Strong Baseline |

> Results obtained on the AffectNet-8 test split (stratified 10%). Training included Class-Weighted Focal Loss, Cosine Annealing, and strong data augmentations (RandAugment).

### Training Progress Summary

| Model | Epochs | Train Acc | Val Acc (Best) |
|---|:---:|:---:|:---:|
| ConvNeXt-Tiny | - | - | *Pending* |
| RegNetY-800MF | - | - | *Pending* |
| ResNet-18 | 21 | 91.8% | **71.90%** |
| EfficientNet-b4 | 22 | 90.1% | **71.58%** |

### VRAM Budget (RTX 4050 6 GB)

| Phase | Component | Peak VRAM |
|---|---|:---:|
| Phase 1 | ConvNeXt-Tiny + Grad-ECLIP | ~2.5 GB |
| Phase 2 | Qwen2.5-0.5B-Instruct (fp16) | ~1.5 GB |

> Both phases fit within 6 GB because the classifier is **unloaded** before the LLM is loaded.

---

## 📁 Project Structure

```
XAI/
├── pyproject.toml                      # Project metadata & all dependencies
├── configs/
│   └── default.yaml                    # Hyperparameters & model config
├── src/
│   ├── pipeline.py                     # End-to-end orchestrator (XAIEmotionPipeline)
│   ├── visualization.py                # Heatmap overlays, landmark drawing, XAI panel
│   ├── face_detection/
│   │   ├── detector.py                 # MediaPipe Tasks FaceLandmarker wrapper
│   │   ├── au_extractor.py             # 468 landmarks → FACS Action Units
│   │   └── models/
│   │       └── face_landmarker.task    # MediaPipe model blob (git-ignored, download manually)
│   ├── emotion/
│   │   ├── model.py                    # Model registry & factory
│   │   ├── convnext.py                 # ConvNeXt-Tiny backbone
│   │   ├── regnet.py                   # RegNetY-800MF backbone
│   │   ├── resnet18.py                 # ResNet-18 baseline
│   │   ├── efficientnet_b4.py          # EfficientNet-B4 baseline
│   │   ├── dataset.py                  # AffectNet / FER2013 image dataloaders
│   │   └── train.py                    # Training loop (AMP, Focal Loss, cosine LR)
│   ├── attention/
│   │   ├── grad_eclip.py               # Grad-ECLIP (channel + spatial importance)
│   │   └── region_parser.py            # Heatmap → 11 semantic region labels
│   └── explainer/
│       ├── prompt_builder.py           # Evidence-based prompt + FACS reference
│       └── vlm_engine.py               # Qwen2.5-0.5B-Instruct text generation
├── scripts/
│   ├── train_classifier.py             # Training CLI (Focal Loss, AMP, auto-resume)
│   ├── evaluate.py                     # Evaluation CLI (accuracy, confusion matrix)
│   ├── extract_landmarks.py            # Pre-extract MediaPipe landmarks to JSON
│   └── demo.py                         # Full XAI demo CLI
├── data/                               # Datasets (git-ignored)
│   └── FER/                            # FER2013 image folders (train/ + test/)
├── checkpoints/                        # Saved model weights (git-ignored)
└── outputs/                            # Results, heatmaps, JSON, XAI panels
```

---

## 🐍 Fallback: pip + venv (without uv)

```bash
python -m venv .venv

# Activate:
source .venv/bin/activate          # Linux / Mac
.venv\Scripts\activate             # Windows

pip install -e ".[dev]"
```

---

## 📝 Research Paper

**Title:** "Beyond Heatmaps: Grounded Textual Explanations for Facial Emotion Recognition via Multimodal Feature Fusion and Vision-Language Models"

### Key Contributions

1. **Novel end-to-end pipeline** — First system combining geometric AU features, Grad-ECLIP attention, and a lightweight LLM into a single grounded explanation pipeline.
2. **Autonomous XAI** — No human expert required; the model generates its own FACS-grounded justification.
3. **Grad-ECLIP for FER** — First application of Grad-ECLIP to facial expression recognition; produces higher-quality heatmaps than standard Grad-CAM on modern vision backbones.
4. **Lightweight deployment** — Runs on a 6 GB consumer GPU via sequential model loading; the LLM (Qwen2.5-0.5B) requires only ~1.5 GB VRAM.
5. **Modern Fast Backbones** — Identifies and integrates lightweight vision models (ConvNeXt-Tiny, RegNetY-800MF) tailored for fast FER without sacrificing expressive power.

### Reference Papers

| Paper | Venue |
|---|---|
| Aly et al. (2023) — *ResNet-50+CBAM for FER* | IEEE Access |
| Zheng et al. (2023) — *POSTER V2 / POSTER++* | ICCV 2023 |
| Zhao et al. (2024) — *Grad-ECLIP* | ICML 2024 |

---

## ⚙️ Hardware Requirements

| Component | Minimum | Tested |
|---|---|---|
| GPU | NVIDIA with 6 GB VRAM | RTX 4050 (6 GB GDDR6) |
| RAM | 16 GB | 16 GB |
| Storage | ~10 GB free | ~15 GB (datasets + weights + outputs) |
| Python | 3.10 | 3.12 |

---

## 📄 License

This project is licensed under the **MIT License**.
