# 🛣️ RoadScan — Road Damage Detection with YOLO (Ultralytics) + Flask

An end-to-end project for detecting road damage (potholes, cracks,
rutting, etc.) from images and videos using YOLO models. It covers:
training and comparing multiple models, hyperparameter tuning, a full
comparative evaluation, and deployment of the best model through a
Flask app that supports **uploading images and videos** for detection,
saving the result as an annotated video.

## Project Overview

| Stage | Description |
|---|---|
| 1. Data | Road Damage dataset from Roboflow (7 classes: alligator, block, crack, edge, longitudinal, pothole, transverse) |
| 2. Training & Comparison | Train and compare 3+ YOLO models (Ultralytics: YOLOv8 / YOLO11) |
| 3. Hyperparameter Tuning | Via `model.tune()` |
| 4. Evaluation | mAP50, mAP50-95, Precision, Recall, real IoU, FPS, model size |
| 5. Deployment | Flask app to upload images/videos and get detection results |

## 1) Setup

```bash
pip install -r deployment/requirements.txt
```

### Getting the dataset

Download the dataset from Roboflow in YOLOv8 format:
```
https://universe.roboflow.com/detelsijalanrusak/road-damage-l1ju7-wbbwq
```
Then update `config/data.yaml` to match the extracted dataset path.

> ⚠️ **Important**: Always check the `data.yaml` content that ships with
> any Roboflow dataset before training. Some Roboflow projects contain
> corrupted class names (README text accidentally used as class labels
> instead of actual damage types). Make sure `names` contains sensible
> damage labels (e.g. `pothole`, `crack`) and not random text.

## 2) Training & Comparison (on Kaggle with GPU)

The recommended approach is to use
`kaggle_notebook/road_damage_yolo_kaggle.ipynb` on Kaggle (free GPU).
Key points:

### a) Run via "Save & Run All (Commit)", not "Edit Mode"

This is **required, not optional**. Running cells manually (Edit Mode)
depends on keeping the browser tab open; any internet drop or closed
tab can stall or complicate tracking the session. "Save & Run All
(Commit)" runs the entire notebook as an independent background job on
Kaggle's servers — you can close the browser completely and you'll get
an automatic email when it finishes (success or failure).

### b) Time budget — a Kaggle session is capped at 12 hours

Before choosing `epochs` and `imgsz`, estimate the expected time:

| Factor | Effect on time |
|---|---|
| `imgsz`: 896 vs 640 | ~2× slower per epoch |
| More `epochs` | Scales linearly with time |
| `TUNE_ITERATIONS × TUNE_EPOCHS` | Each tuning trial is its own short training run; keep `TUNE_IMG_SIZE` smaller than the final `IMG_SIZE` |
| `cache="ram"` | Speeds up every epoch by loading images into memory once |
| `multi_scale=True` | Roughly doubles time per epoch; enable only if your budget allows it |

**Rule of thumb**: tuning should always be cheaper than final training
(smaller imgsz, fewer epochs) — its job is just to point toward good
hyperparameters, not to be a full training run.

### c) Safe settings for a single session (one model within ~12 hours)

```python
MODELS = ["yolov8s.pt"]
IMG_SIZE = 768                     
FINAL_EPOCHS = 100               
PATIENCE_EARLY_STOP = 20       

BASELINE_EPOCHS = 0               
TUNE_ITERATIONS = 0                
TUNE_EPOCHS = 0
TUNE_IMG_SIZE = 640
IOU_SAMPLE = 80
```

The project requires **at least 3 YOLO models** for comparison — repeat
this setup across separate sessions (one model per session) to avoid
exceeding the 12-hour limit, then merge the three results tables
manually at the end.

### d) `train_final` — important speed settings

```python
 model.train(
        data=data_yaml, 
        epochs=epochs, 
        imgsz=imgsz,
        batch=8,                        
        cache=False,                   
        patience=PATIENCE_EARLY_STOP,   
        amp=True, 
        seed=42, 
        plots=True,
        project=project, 
        name=model_name.replace(".pt", ""), 
        exist_ok=True,
        **DEFAULT_HYP,
    )
```

> ⚠️ **Common error**: a fixed `batch=16` can fail with
> `ValueError: Expected more than 1 value per channel when training`
> if the number of training images doesn't divide evenly by 16 (leaving
> a last batch of size 1, which BatchNorm can't handle mathematically).
> Fix: use `batch=0.80` (a float between 0-1 = fraction of GPU memory)
> instead of a fixed integer.

### e) Comparison outputs

After running, you'll find in `results/`:

| File | Content |
|---|---|
| `comparison_table.csv` | mAP50, mAP50-95, precision, recall, mean_IoU, FPS, model_size_MB, params_M, GFLOPs, deployment_efficiency |
| `comparison_map_chart.png` | mAP50 vs mAP50-95 chart |
| `comparison_precision_recall.png` | Precision vs Recall chart |
| `comparison_speed_vs_accuracy.png` | Bubble chart: FPS vs mAP50-95 |
| `comparison_efficiency_chart.png` | Models ranked by deployment efficiency |
| `comparative_analysis.md` | Auto-generated report discussing the results |

The winning model (by `deployment_efficiency` by default) is
automatically copied to `weights/best_overall/best.pt`.

## 3) Moving `best.pt` to the deployment app

Download `best.pt` from Kaggle's **Output** tab and place it directly
next to `deployment/app.py`.

> ⚠️ **Important download note**: a `.pt` file is internally a ZIP
> archive (PyTorch's native format). Some browsers/OSes auto-extract it
> on download, leaving you with a **folder** instead of a single file
> (containing files like `data.pkl`, `byteorder`, `data/`). If this
> happens:
> - Make sure you're clicking the direct download icon for `best.pt`
>   itself in the Output tab, not downloading the whole `weights/`
>   folder as an archive
> - If it still happens, re-zip the extracted folder's contents into a
>   new archive (`ZIP_STORED`, uncompressed) with a `.pt` extension,
>   placing all files under one internal path (e.g. `archive/...`) so
>   it's compatible with `torch.load`

Verify the file is valid before running the app:

```python
from ultralytics import YOLO
model = YOLO("best.pt")
print(model.names)   # should show the correct 7 classes
```

## 4) Running the deployment app (Flask)

```bash
cd deployment
pip install -r requirements.txt
python app.py
```

Open your browser at: `http://127.0.0.1:5000`

### Features

- **Image upload**: instant detection with drawn bounding boxes,
  class labels, and confidence scores
- **Video upload**: processes the video frame-by-frame through the
  model and saves an annotated copy to `outputs/`, with a link to
  view/download it from the page
- Adjustable **Confidence** and **IoU (NMS)** thresholds directly from
  the UI
- Optional **"High Accuracy" mode** that enables Test-Time
  Augmentation (`augment=True`)

### Upload limits

Check `MAX_CONTENT_LENGTH` in `app.py` if you need to upload videos
larger than the default limit; longer videos take longer to process
since every frame is passed through the model individually.

## 5) Getting a sample video for testing

Free, royalty-free video sources for road damage (potholes, cracks):

- **Pexels**: `pexels.com/search/videos/potholes` — includes a
  "Duration" filter to pick clips of a specific length (e.g. 30-60s)
- **Pixabay**: search "road potholes" in the video section

Pick a clip with a ground-level camera angle (from a car or on foot),
not aerial, since that matches the training data's perspective.

## 6) Improving accuracy (mAP / Precision / Recall / IoU)

### a) Loss weights (focus on tighter box alignment)

```python
DEFAULT_HYP = dict(
    box=10.0,        # Higher box-loss weight → directly improves IoU
    cls=0.4,
    dfl=1.6,
    ...
)
```

### b) Balanced augmentation

Road damage (cracks/potholes) is small and sensitive to heavy
distortion — tone down `mixup`, `copy_paste`, and `shear` rather than
disabling them completely, to preserve natural damage shapes during
training.

### c) TTA + Weighted Boxes Fusion (inference time, no retraining)

`inference/tta_wbf_ensemble.py` applies:
- **TTA**: runs the model on augmented copies of the image (different
  scales + flip)
- **WBF**: fuses overlapping boxes with a confidence-weighted average
  instead of discarding them like standard NMS, producing a final box
  more tightly aligned with ground truth

### d) A realistic note on IoU

An IoU of 1.0 means a perfect geometric match, which is practically
unverifiable even with a great model, due to natural noise in human
box annotation and the irregular shape of some damage types (e.g.
cracks). A realistic target is getting as close as possible (0.85+),
not a literal 1.0.

## Requirements (deployment/requirements.txt)

```
ultralytics>=8.3.0
torch>=2.2.0
torchvision>=0.17.0
opencv-python>=4.9.0
Flask>=3.0.0
Pillow>=10.2.0
numpy>=1.26.0
```
