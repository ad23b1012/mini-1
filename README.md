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
│  Branch 1: AU Extractor          Branch 2: POSTER V2           │
│  (geometric distances → FACS)    (IR-50 backbone)              │
│                                       ↓                        │
│  468 landmarks → dense projection → 49 query tokens            │
│                                       ↓                        │
│                           Window Cross-Attention (×2)          │
│                                       ↓                        │
│                               Emotion Logits (7 classes)       │
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
| Emotion classifier | **POSTER V2** (IR-50 + Window Cross-Attention) | 7-class FER2013 labels |
| Baseline classifier | ResNet-50 + CBAM | For ablation comparison |
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

### FER2013 (Primary Training Dataset)

The training script expects an **image folder** structure (not the raw CSV):

```
data/FER/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── angry/
    └── ...
```

**Option 1 — Kaggle CLI:**
```bash
uv run kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/FER/
```

**Option 2 — Manual download:**  
Visit [kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013) and place the extracted image folders in `data/FER/`.

### RAF-DB (Optional)

Download from [the RAF-DB project page](http://www.whdeng.cn/RAF/model1.html) and place in `data/RAF-DB/`.

---

## 🚀 Usage

> **Always prefix commands with `uv run`** when using the `uv` managed environment.

### 1. Train the Emotion Classifier

```bash
# Primary model — POSTER V2 with MediaPipe landmark guidance
uv run python scripts/train_classifier.py \
    --model poster_v2 \
    --dataset fer2013 \
    --data-path data/FER \
    --epochs 50 \
    --batch-size 32

# With pre-extracted landmarks (improves cross-attention quality)
uv run python scripts/train_classifier.py \
    --model poster_v2 \
    --dataset fer2013 \
    --data-path data/FER \
    --landmarks-file data/fer2013_train_landmarks.json \
    --epochs 50 \
    --batch-size 32

# Baseline model — ResNet-50 + CBAM (ablation study)
uv run python scripts/train_classifier.py \
    --model resnet50_cbam \
    --dataset fer2013 \
    --data-path data/FER \
    --epochs 50
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
    --model poster_v2 \
    --dataset fer2013 \
    --data-path data/FER \
    --checkpoint checkpoints/poster_v2_best.pth \
    --split test
```

Outputs a per-class classification report and confusion matrix PNG to `outputs/`.

### 3. Extract Facial Landmarks

Pre-extract landmarks from all training images for higher-quality POSTER V2 training:

```bash
uv run python scripts/extract_landmarks.py --image path/to/face.jpg
```

### 4. Run the Full XAI Pipeline

```bash
# Full pipeline: face detection → emotion → attention → Qwen explanation
uv run python scripts/demo.py --image path/to/face.jpg

# Skip the LLM (faster — classification + attention map only)
uv run python scripts/demo.py --image path/to/face.jpg --no-explanation

# Use a trained checkpoint and a specific attention method
uv run python scripts/demo.py \
    --image path/to/face.jpg \
    --checkpoint checkpoints/poster_v2_best.pth \
    --attention grad_eclip

# Use the baseline classifier instead
uv run python scripts/demo.py \
    --image path/to/face.jpg \
    --model resnet50_cbam \
    --checkpoint checkpoints/resnet50_cbam_best.pth
```

All output files (JSON result, face crop, XAI panel PNG) are saved under `outputs/<image_name>/`.

---

## 🔄 Pipeline Flow (Step-by-Step)

```
Step 1  →  MediaPipe FaceLandmarker detects face + extracts 468 landmarks
Step 2  →  AUExtractor computes geometric distances → activates FACS Action Units
Step 3  →  POSTER V2 classifies emotion (landmark queries + IR-50 visual features)
Step 4  →  Grad-ECLIP generates attention heatmap from backward gradients
Step 5  →  RegionParser maps heatmap intensity → semantic facial regions
Step 6  →  [VRAM swap] Classifier unloaded → Qwen2.5-0.5B-Instruct loaded
Step 7  →  PromptBuilder assembles evidence prompt (emotion + AUs + regions + FACS)
Step 8  →  Qwen generates the textual explanation (greedy decode, ≤150 tokens)
Step 9  →  Visualization: landmark overlay + heatmap overlay + combined XAI panel saved
```

---

## 📊 Results

### Emotion Classification Accuracy

| Model | FER2013 (Test) | RAF-DB |
|---|:---:|:---:|
| Aly et al. (ResNet-50+CBAM, 2023) | 73.43% | 87.62% |
| ResNet-50+CBAM *(ours, reproduced)* | TBD | TBD |
| **POSTER V2 *(ours, 46 epochs)*** | **62.44%** | TBD |

> FER2013 test split (7,178 images). Training ran for 46 epochs with Class-Weighted Focal Loss, AMP, and cosine LR schedule. Best validation accuracy reached **62.45%** at epoch 36.

### Per-Class Performance — POSTER V2 on FER2013 Test Set

| Emotion | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| Angry | 56.4% | 47.1% | 51.3% | 958 |
| Disgust | 37.8% | 64.0% | 47.5% | 111 |
| Fear | 47.4% | 35.2% | 40.4% | 1,024 |
| **Happy** | **86.8%** | **81.7%** | **84.2%** | 1,774 |
| Sad | 49.5% | 56.4% | 52.7% | 1,247 |
| **Surprise** | **67.0%** | **81.9%** | **73.7%** | 831 |
| Neutral | 57.8% | 62.1% | 59.9% | 1,233 |
| **Weighted Avg** | **62.6%** | **62.4%** | **62.1%** | **7,178** |

> **Notes:** Happy and Surprise are the strongest classes due to clear geometric AU signals. Disgust (n=111) and Fear suffer from extreme class imbalance despite Focal Loss weighting — a known FER2013 challenge. Training without a pre-trained IR-50 backbone (trained from scratch) is the primary bottleneck; loading ArcFace weights would likely push accuracy toward 67–70%.

### Training Curve Summary

| Phase | Epochs | Train Acc | Val Acc |
|---|:---:|:---:|:---:|
| Warmup (cosine ramp) | 1–5 | 13% → 37% | 19% → 36% |
| Main training | 6–36 | 41% → 62% | 43% → **62.45%** ← best |
| Late convergence | 37–46 | 67% → 70% | 59% → 59% (slight overfit) |

### VRAM Budget (RTX 4050 6 GB)

| Phase | Component | Peak VRAM |
|---|---|:---:|
| Phase 1 | POSTER V2 + Grad-ECLIP | ~2.5 GB |
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
│   │   ├── model.py                    # POSTER V2 (IR-50 + Window Cross-Attention)
│   │   ├── baseline_resnet_cbam.py     # ResNet-50+CBAM baseline
│   │   ├── dataset.py                  # FER2013 / RAF-DB image folder dataloaders
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
3. **Grad-ECLIP for FER** — First application of Grad-ECLIP to facial expression recognition; produces higher-quality heatmaps than standard Grad-CAM on transformer-based backbones.
4. **Lightweight deployment** — Runs on a 6 GB consumer GPU via sequential model loading; the LLM (Qwen2.5-0.5B) requires only ~1.5 GB VRAM.
5. **Landmark-guided cross-attention** — POSTER V2 is extended to consume real MediaPipe (468 × 2) geometric coordinates as structural query tokens, replacing the original fixed token approach.

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
