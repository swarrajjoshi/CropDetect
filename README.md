# CropDetect — Crop Disease & Pest Diagnosis Expert System

A working Python case study for IIS / IE. **Scope: tomato leaves, two fungal diseases, spider-mite damage, and healthy leaves.** It combines 12 explicit visual features, trained ML, a separate binary pest model, and an explainable recommendation rule engine.

## Run the completed demo

Double-click `start.bat`, or run this from PowerShell:

```powershell
.\start.ps1
```

Open http://127.0.0.1:5000. Choose one of the four held-out sample images or upload a single tomato leaf. Click **Analyze leaf**. Inference works offline after installation and training; optional Google fonts fall back to local fonts.

## Verify before deployment

Run both the automated suite and the broader deployment smoke test:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/verify_deployment.py --per-class 5
```

The smoke test exercises the public pages and APIs, the four bundled demo images, 20 held-out test images across all supported classes, PNG/WebP uploads, feature/probability invariants, recommendations, and invalid/oversized upload handling. It writes the detailed result to `artifacts/verification_results.json`.

## Deploy

The repository includes `render.yaml`, a `Procfile`, and a `Dockerfile`. For Render, push the project to a GitHub/GitLab/Bitbucket repository and create a Blueprint from `render.yaml`. The service installs `requirements.txt`, starts Gunicorn on Render's `$PORT`, and checks `/api/health`. The four demo images are copied into `static/samples`, so the deployed demo does not require the ignored `data/raw` directory.

For another container platform:

```powershell
docker build -t cropdetect .
docker run --rm -p 8000:8000 -e PORT=8000 cropdetect
```

Open `http://127.0.0.1:8000`. Keep `artifacts/model.joblib`, `metrics.json`, `sample_predictions.json`, the report images, and `static/samples` in the deployed source bundle.

## Reproduce from scratch

Python 3.12 is the tested runtime. With Python on PATH:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_dataset.py --per-class 400
.\.venv\Scripts\python.exe scripts/train.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe app.py
```

The supplied local `.venv` is ready on this machine. It is not portable: recreate it on another computer. Training artifacts are trusted local joblib files; never load model files from strangers.

## Technical pipeline

1. **Dataset:** deterministic balanced sample from the original [PlantVillage repository](https://github.com/spMohanty/PlantVillage-Dataset), not synthetic images. `data/manifest.json` records pinned repository revision, source URLs, byte and pixel SHA-256 hashes, source class counts, and selected filenames. Downloaded originals live in `data/raw`. The dataset authors retain their rights; review upstream usage terms before redistributing images. Reference: [Hughes & Salathé, 2015](https://arxiv.org/abs/1511.08060).
2. **Data processing:** decode RGB pixels, remove exact pixel duplicates, reject conflicting duplicate labels, and stratify approximately 70%/15%/15% using seed 42. `artifacts/splits.json` preserves membership. The exact post-deduplication counts and rejected images are in `artifacts/metrics.json`.
3. **Preprocessing:** EXIF orientation, aspect-preserving resize/pad to 160×160, median denoising, saturation/value foreground heuristic, largest filled contour, CLAHE illumination enhancement. The normalized RGB array is in [0,1]; color features are explicitly scaled and the SVM standardizes all features using training statistics. Background estimates are heuristics, not learned segmentation or lesion ground truth.
4. **Feature extraction:** 12 scalars; details below. `artifacts/features.csv` contains actual values and split/augmentation flags.
5. **Training:** Random Forest versus standardized RBF SVM, selected by validation macro F1. One flipped/dimmed variant per training image only. Validation and test images are never augmented or fitted. Selected model stays fitted on training data; the untouched test set is evaluated once. A separately trained binary RBF SVM detects spider-mite damage versus all other supported classes.
6. **Prediction:** the uploaded image follows exactly the training feature path. Return class probabilities, candidate diagnosis, and binary pest probability. SVC probability estimation is fitted on training data; it is not a field calibration guarantee. Its internal calibration folds can contain related augmented training images; the external validation/test partitions remain isolated.
7. **Knowledge and reasoning:** forward-chaining rules check mask coverage, confidence, and model agreement. Uncertain cases request review and withhold class-specific treatment. Supported candidates retrieve a versioned, source-linked curated knowledge base through the same service used by the recommendation REST API.
8. **Prevention:** practical class-specific cultural controls, presented alongside source attribution and limits on chemical guidance.

## Exactly 12 features

Computed on enhanced foreground pixels; HSV uses OpenCV H∈[0,179], S,V∈[0,255]. Color fractions are descriptive proxies, not biological lesion measurements.

| # | Feature | Definition / meaning |
|---|---|---|
| 1 | hue_mean | Mean H / 179; dominant color |
| 2 | saturation_mean | Mean S / 255; color intensity |
| 3 | value_mean | Mean V / 255; brightness |
| 4 | hue_std | Std(H) / 179; color variation; linear hue statistic |
| 5 | saturation_std | Std(S) / 255; pigmentation variation |
| 6 | green_fraction | Foreground fraction with 35 ≤ H ≤ 90 |
| 7 | yellow_fraction | Foreground fraction with 18 ≤ H < 35; chlorosis proxy |
| 8 | brown_fraction | H < 25, S > 65, V < 180; dark/brown tissue proxy |
| 9 | glcm_contrast | Σ P(i,j)(i−j)²; local intensity variation |
| 10 | glcm_homogeneity | Σ P(i,j)/(1+(i−j)²); local similarity |
| 11 | glcm_energy | sqrt(Σ P(i,j)²); texture regularity |
| 12 | leaf_solidity | Contour area / convex hull area; outline compactness |

Texture uses 32 gray levels, symmetric normalized co-occurrence matrices, offset distance 1 at four directions. Both pixels must be inside the leaf mask; background pairs are excluded. Texture properties are averaged over directions.

## Outputs and evaluation

- `artifacts/model.joblib`: trained selected classifier, binary pest classifier, feature schema, review threshold.
- `artifacts/metrics.json`: accuracy; per-class and macro/weighted precision, recall, F1, support; validation comparison; confusion matrices for disease and pest models; counts and exclusions.
- `artifacts/confusion_matrix.png`: actual held-out multiclass predictions.
- `artifacts/preprocessing.png`: original/enhanced/mask examples for every class.
- `artifacts/sample_predictions.json`: real test predictions, including any errors.
- `artifacts/features.csv` and `artifacts/splits.json`: auditable experiment data.

Never substitute published paper accuracy for this model's accuracy. Read the generated report or UI for the actual experiment. The test metrics cover all predictions, not just those accepted by the rule engine; selective accuracy has not been evaluated.

## API

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/analyze` | Multipart field `image`; full diagnostic response |
| GET | `/api/recommendations/<label>` | Curated class guidance; educational lookup independent of a diagnosis |
| GET | `/api/metrics` | Actual training/evaluation report |
| GET | `/api/health` | Model availability and supported labels |
| GET | `/api/samples` | Held-out sample metadata |

Labels: `early_blight`, `septoria`, `spider_mites`, `healthy`. Missing/invalid images return 400, uploads exceeding 8 MB return 413, and absent model artifacts return 503. Images are processed in memory and not saved by the upload endpoint. The server binds to localhost, with debug disabled.

```powershell
curl.exe -F "image=@path-to-leaf.jpg" http://127.0.0.1:5000/api/analyze
```

## Module 4 connection

Knowledge representation: facts (ML candidate, probability, mask coverage, pest evidence), a versioned knowledge base, and explicit IF/THEN rules. Forward chaining derives whether guidance may be shown. Uncertainty is represented by model probabilities and abstention; it is not proof of a diagnosis. The inference trace explains the rules, not the internal causal reasoning of the classifier. Business constraints include limited classes, no paid API dependency, low compute, offline operation, treatment registration differences, and human confirmation. This implementation does not claim to implement every Module 4 topic such as resolution, full first-order theorem proving, or Bayesian networks.

## Limits you should state in the presentation

- The pest component is **ML classification of spider-mite damage**, not object detection, counting, or a general insect recognizer. Spider mites are arachnids. If the examiner specifically requires insect species or bounding boxes, a separately annotated insect dataset and detector are needed.
- The system supports only the four listed tomato conditions. Field backgrounds, other crops, mixed infections, and unknown diseases can be misclassified confidently. Mask/confidence checks are not a validated out-of-distribution detector.
- PlantVillage is collected under controlled conditions. Exact duplicate removal does not prevent all near-duplicate or same-plant leakage. No plant-level identifiers are available in this subset; no field test was conducted.
- Threshold 0.75 is selected as the lowest tested cutoff that reached at least 95% selective accuracy on the validation set after binary/multiclass agreement. It remains an educational operating point, not an agronomically validated threshold. Agreement is not independent confirmation because both models share features and training images.
- Recommendations are **a curated knowledge base exposed through our REST API**, not a live commercial pesticide API. US extension sources are educational references, not approval to use a pesticide in India. No application rates, brand endorsements, or automatic chemical prescriptions are provided.

## Six-person presentation

See `docs/PRESENTATION_GUIDE.md` for ownership, a speaking sequence, demonstration steps, and likely questions. The source modules are deliberately separated by stage.
