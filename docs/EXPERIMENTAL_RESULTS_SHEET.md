# Experimental Results Sheet — Verified Project Values

Generated from the implemented CropDetect code and artifacts on 2026-10-04. Values marked **not available** were not produced by this project. The implementation is a four-class tomato-leaf system; it is not the proposed 15-class MobileNet system.

## 1. Dataset

| Item | Verified value | Evidence |
|---|---:|---|
| Downloaded/selected images | 1,600 | `data/manifest.json`, `artifacts/metrics.json` |
| Supported classes | 4 | `cropdetect/config.py` |
| Classes | `early_blight`, `healthy`, `septoria`, `spider_mites` | `cropdetect/config.py` |
| Selected per class | 400 each | `data/manifest.json` |
| Available in pinned source folders | Early blight 1,000; healthy 1,591; Septoria 1,771; spider mites 1,676; total 6,038 | `artifacts/metrics.json` → `source_counts` |
| Original train/validation/test counts | 1,120 / 240 / 240 | `artifacts/metrics.json` → `split_counts` |
| Per-class train/validation/test | 280 / 60 / 60 | `artifacts/metrics.json` |
| Split ratio | 70% / 15% / 15% | `scripts/train.py` lines 74–75 |
| Random seed | 42 | `cropdetect/config.py` |
| Stratified | Yes, for both split operations | `scripts/train.py` lines 74–75 |
| Duplicate handling | Exact decoded-pixel SHA-256 duplicates removed before splitting; 0 removed in this selected subset | `scripts/train.py`, `artifacts/metrics.json` |
| Quality filtering | Images that fail usable-mask feature extraction are rejected; 0 rejected | `scripts/train.py`, `artifacts/metrics.json` |

Training-only augmentation produces 2,240 training feature rows. Validation and test remain 240 original images each.

## 2. Preprocessing

| Item | Verified value |
|---|---|
| Input tensor/image size | 160 × 160 RGB |
| Orientation | EXIF transpose, then RGB conversion |
| Resize | Aspect-preserving `PIL.ImageOps.pad`; default bicubic resampling; padding colour `(128,128,128)`; centered |
| Noise filter | 3 × 3 median filter |
| Bilateral filter | **Not implemented** |
| Foreground handling | HSV `S > 38` and `V > 25`; 7 × 7 morphological closing; largest external contour filled as mask |
| CLAHE | LAB L-channel; clip limit 2.0; tile grid 8 × 8 |
| Normalization | Enhanced RGB cast to float32 and divided by 255, giving [0,1] |
| Augmentation | Exactly one deterministic training variant per source: horizontal reflection followed by brightness factor 0.9 |
| Augmentation ranges | **Not available**; no random range is used |
| Train/validation/test difference | Core preprocessing is identical; only training receives augmentation |

Evidence: `cropdetect/preprocessing.py` and `scripts/train.py`.

## 3. Model

The selected multiclass model is a scikit-learn pipeline:

1. `StandardScaler` fitted on training features.
2. `SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)`.
3. Input is the 12-value handcrafted visual feature vector in `cropdetect/features.py`.

The multiclass SVM has 739 support vectors: 190 early blight, 153 healthy, 186 Septoria and 210 spider-mite damage. Its fitted dual coefficient array has shape `(3,739)` and six pairwise intercepts.

A separate binary pest pipeline uses `StandardScaler` plus class-balanced RBF SVC with `C=10`, `probability=True`, and seed 42. It has 475 support vectors: 328 non-pest and 147 pest.

| Requested neural-network field | Verified value |
|---|---|
| Pretrained weights/backbone | Not applicable; no neural network |
| Frozen/fine-tuned backbone | Not applicable |
| Trainable/total parameter count | Not applicable for this kernel SVM; no neural parameter count |
| Classification head | Not applicable |
| Activation functions/dropout | Not applicable |
| Loss, optimizer, LR, weight decay | Not exposed/used as neural-training settings |
| Batch size/epochs/scheduler/early stopping/best epoch | Not applicable; SVC is not epoch-trained |
| Label smoothing | Not implemented |

Evidence: `scripts/train.py`, `artifacts/model.joblib`.

## 4. Training and Validation Results

Post-hoc evaluation of the fitted selected model on the stored feature rows:

| Split | Rows | Accuracy | Macro F1 | Negative log likelihood |
|---|---:|---:|---:|---:|
| Training, including augmentation | 2,240 | 0.9741071428571428 | 0.9740998254846845 | 0.07980722021361886 |
| Validation | 240 | 0.8916666666666667 | 0.8916268770310594 | 0.21432899456648227 |
| Test | 240 | 0.9166666666666666 | 0.9165617371311543 | 0.23201875209338557 |

Candidate validation macro F1: Random Forest `0.8585277561671649`; RBF SVM `0.8916268770310594`.

Final/best epoch accuracy, epoch losses, best epoch, and epoch-by-epoch history are **not available** because the selected SVM has no epoch loop. No training-history file exists. The NLL values above are post-hoc probability metrics, not an epoch loss history.

## 5. Held-Out Test Results

| Metric | Exact value |
|---|---:|
| Accuracy | 0.9166666666666666 |
| Test loss | **Not available** |
| Post-hoc negative log likelihood | 0.23201875209338557 |
| Macro precision | 0.9178927453270302 |
| Macro recall | 0.9166666666666666 |
| Macro F1 | 0.9165617371311543 |
| Weighted precision | 0.9178927453270302 |
| Weighted recall | 0.9166666666666666 |
| Weighted F1 | 0.9165617371311543 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Early blight | 0.9137931034482759 | 0.8833333333333333 | 0.8983050847457628 | 60 |
| Healthy | 0.9344262295081968 | 0.95 | 0.9421487603305785 | 60 |
| Septoria | 0.8769230769230769 | 0.95 | 0.912 | 60 |
| Spider-mite damage | 0.9464285714285714 | 0.8833333333333333 | 0.9137931034482759 | 60 |

Evidence: `artifacts/metrics.json`.

## 6. Confusion Matrix

Rows are actual classes and columns are predicted classes in this order: early blight, healthy, Septoria, spider-mite damage.

```text
[[53, 0, 5, 2],
 [ 2,57, 0, 1],
 [ 1, 2,57, 0],
 [ 2, 2, 3,53]]
```

| One-vs-rest class | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| Early blight | 53 | 5 | 7 | 175 |
| Healthy | 57 | 4 | 3 | 176 |
| Septoria | 57 | 8 | 3 | 172 |
| Spider-mite damage | 53 | 3 | 7 | 177 |

Correct predictions: 220. Incorrect predictions: 20. The largest directed confusion is early blight → Septoria, 5 images. The largest combined class pair is early blight/Septoria, 6 errors across both directions.

Numerical source: `artifacts/metrics.json`. Figure: `artifacts/confusion_matrix.png`.

## 7. Confidence and Calibration

Temperature scaling is **not implemented**; temperature `T` and its calibration dataset are **not available**. SVC `probability=True` uses its internal probability fitting.

The implemented review threshold is **0.75** in `cropdetect/config.py`. It is the lowest tested threshold that reached at least 95% selective validation accuracy after requiring pest-model agreement.

| Test-set confidence result | Exact value |
|---|---:|
| At or above 0.75 | 220 / 240 = 0.9166666666666666 |
| Below 0.75 | 20 / 240 = 0.08333333333333333 |
| Accepted after pest-model agreement | 220 / 240 |
| Accuracy among accepted predictions | 210 / 220 = 0.9545454545454546 |
| Multiclass Brier score | 0.12349033764086487 |
| Negative log likelihood | 0.23201875209338557 |
| ECE, 10 equal-width bins | 0.03994078528148559 |

These are held-out, post-hoc measurements, not field calibration. Ten wrong predictions still pass the gate, including high-confidence early-blight/Septoria errors. Confidence does not guarantee correctness.

Evidence: `artifacts/metrics.json` → `confidence_policy`.

## 8. Real Example Predictions

| Input filename | True | Predicted | Confidence | Gate | Correct |
|---|---|---|---:|---|---|
| `fbb79594-419e-4bb7-9afa-6b9c0ba89033___RS_Erly.B 7644.JPG` | Early blight | Early blight | 0.9999975200082976 | Passed | Yes |
| `fc1e8e1c-5939-4ad9-b87e-2cbb77cef418___RS_HL 0536.JPG` | Healthy | Healthy | 0.9999345346782683 | Passed | Yes |
| `4f45a4b4-342b-4422-87b9-8386a6801ba1___JR_Sept.L.S 2566.JPG` | Septoria | Septoria | 0.9999996271526844 | Passed | Yes |
| `4ad85b66-43f0-476f-aee4-a320e793a42f___Com.G_SpM_FL 8645.JPG` | Spider-mite damage | Spider-mite damage | 0.9999999098691312 | Passed | Yes |
| `c83d0821-505b-4270-acd7-1a5c79f671b4___RS_Erly.B 7629.JPG` | Early blight | Septoria | 0.9932636824194128 | Passed | No |

The four correct examples are stored in `artifacts/sample_predictions.json`; their deployment copies are `static/samples/0.jpg` through `3.jpg`. Potato late blight, apple disease and corn disease examples are **not available** because these classes do not exist in this project.

## 9. Runtime and Deployment

Measured locally on Windows 11, an AMD64 Family 25 Model 124 CPU, 12 logical CPUs, CPU inference only:

| Item | Measured value |
|---|---:|
| Requests measured | 20 warm complete `/api/analyze` requests |
| Mean response time | 21.331514998018974 ms |
| Median | 19.54394999484066 ms |
| p95 | 34.36859999783337 ms |
| Minimum / maximum warm | 15.085199993336573 / 37.3299000057159 ms |
| Cold first request | 2715.6981000007363 ms |
| Model file | 169,104 bytes |
| RAM / VRAM | **Not available** |
| Python | 3.12.14 |
| Flask | 3.1.3 |
| scikit-learn | 1.9.1 |
| scikit-image | 0.26.0 |
| OpenCV headless | 5.0.0.93 |
| NumPy | 2.5.3 |
| Pillow | 12.3.0 |
| joblib | 1.6.0 |

The timing includes decoding, preprocessing, 12-feature extraction, two model calls, reasoning, three PNG encodes and JSON serialization through Flask's test client. It excludes network latency. Source: `artifacts/runtime_metrics.json`.

Production files: `render.yaml`, `Procfile`, `Dockerfile`, `.dockerignore`. The local updated app is running at `http://127.0.0.1:5000/`.

## 10. Available Figures and Reports

| Needed item | Exact file |
|---|---|
| Training accuracy/loss graph | **Not available** |
| Validation accuracy/loss graph | **Not available** |
| Confusion matrix | `artifacts/confusion_matrix.png` |
| Dataset distribution figure | **Not available**; numerical counts are in `artifacts/metrics.json` |
| Sample predictions | `artifacts/sample_predictions.json`, `static/samples/0.jpg`–`3.jpg` |
| UI screenshot | **Not available** |
| Architecture/model diagram | **Not available**; textual architecture is in `README.md` and `docs/CASE_STUDY.md` |
| Evaluation report | `artifacts/metrics.json` |
| Deployment verification | `artifacts/verification_results.json` |
| Runtime benchmark | `artifacts/runtime_metrics.json` |
| Full technical report | `CropDetect_Expert_System_Project_Report.docx` |

## 11. Draft Consistency Check

| Claim | Actually implemented? | Actual value | Evidence |
|---|---|---|---|
| 16,846 images | No | 1,600 selected/downloaded; 6,038 available across the four source folders | `data/manifest.json`, `artifacts/metrics.json` |
| 15 classes | No | 4 tomato classes | `cropdetect/config.py` |
| 70:15:15 split | Yes | 1,120 / 240 / 240 originals | `scripts/train.py`, `artifacts/metrics.json` |
| MobileNetV3-Large | No | StandardScaler + RBF SVM; separate binary RBF SVM | `scripts/train.py`, `artifacts/model.joblib` |
| 224×224 input | No | 160×160 RGB | `cropdetect/preprocessing.py` |
| Bilateral filtering | No | 3×3 median filter | `cropdetect/preprocessing.py` |
| CLAHE | Yes | LAB L-channel, clip 2.0, 8×8 tiles | `cropdetect/preprocessing.py` |
| Temperature scaling T=1.20 | No | No temperature scaling; T not available | `scripts/train.py`, `artifacts/metrics.json` |
| Confidence threshold 0.75 | Yes | 0.75 | `cropdetect/config.py`, `artifacts/model.joblib` |
| AdamW | No | Not applicable to SVM | `scripts/train.py` |
| LR 1e-3 | No | Not applicable | `scripts/train.py` |
| Weight decay 1e-4 | No | Not applicable | `scripts/train.py` |
| Label smoothing 0.1 | No | Not implemented | `scripts/train.py` |
| Cosine annealing | No | Not implemented | `scripts/train.py` |
| 30 epochs | No | No epochs; SVC fit | `scripts/train.py` |
| 93.85% example confidence | No matching stored example | Not available | `artifacts/sample_predictions.json` |
| 94.62% example confidence | No matching stored example | Not available | `artifacts/sample_predictions.json` |
| ~118 ms latency | No | Warm mean 21.331514998018974 ms; cold first request 2715.6981000007363 ms in the measured local setup | `artifacts/runtime_metrics.json` |

## FINAL VERIFIED VALUES FOR TECHNICAL REPORT

- Dataset: 1,600 selected PlantVillage tomato images, 400 per class, four classes.
- Classes: early blight, healthy, Septoria leaf spot, spider-mite damage.
- Split: 1,120 train / 240 validation / 240 test; 70% / 15% / 15%; stratified; seed 42.
- Augmented training rows: 2,240; validation/test are not augmented.
- Input: 160×160 RGB; bicubic aspect-preserving pad, 3×3 median filter, HSV foreground mask, 7×7 closing, largest contour, LAB CLAHE 2.0 with 8×8 tiles, divide by 255.
- Features: exactly 12 handcrafted colour, texture and shape values.
- Model: StandardScaler + RBF SVC (`C=10`, `gamma='scale'`, probability enabled); separate class-balanced binary RBF SVC pest model.
- Validation: accuracy 0.8916666666666667; macro F1 0.8916268770310594.
- Test: accuracy 0.9166666666666666; macro precision 0.9178927453270302; macro recall 0.9166666666666666; macro F1 0.9165617371311543.
- Correct/incorrect test predictions: 220 / 20.
- Binary pest test accuracy: 0.9583333333333334.
- Review threshold: 0.75; accepted test predictions 220/240; accepted accuracy 0.9545454545454546.
- Temperature scaling: not implemented.
- Warm complete API mean: 21.331514998018974 ms over 20 measured requests; cold first request 2715.6981000007363 ms.
- Model size: 169,104 bytes.
- Deployment verification: 6 automated tests passed; 39/39 smoke checks passed across 20 held-out images and valid/invalid upload cases.
