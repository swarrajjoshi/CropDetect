# Six-person technical case study guide

## P1 — Problem and system overview

Explain why symptoms can overlap and why a label alone is insufficient. Show the full flow: image → preprocessing → 12 measurements → two trained models → facts → inference rules → curated treatment and prevention. State that this is a four-class tomato case study and that the pest is spider mite. Connect to Module 4: knowledge representation, forward chaining, uncertainty, explanation, and business constraints.

## P2 — Dataset and data processing

Open `data/manifest.json` and `artifacts/metrics.json`. Read out the actual source counts, selected subset counts, removed duplicates and train/validation/test counts. Explain seeded random sampling, pixel checksums, stratification, and why the test set must stay untouched. Demonstrate `artifacts/splits.json`. Do not claim the full PlantVillage dataset was trained.

## P3 — Preprocessing

Open `artifacts/preprocessing.png`. Explain aspect-preserving resize, median filter, foreground mask, CLAHE, and numeric normalization. Point out an imperfect mask if present. Show `cropdetect/preprocessing.py`. Explain that augmentation is applied only to training sources after the split. Brightness variation and reflection simulate limited capture variation; neither proves field robustness.

## P4 — Feature extraction and 12 features

Use the README feature table and `cropdetect/features.py`. Show one row from `artifacts/features.csv`. Explain three feature families: color/proxy damage (8), texture (3), shape (1). Explain that a GLCM counts adjacent intensity pairs inside the leaf, not the black background. These features are actual numbers used by the models, not decorative dashboard values. Brown/yellow fractions are heuristic proxies, not annotated lesion area.

## P5 — ML classification, diagnosis and evaluation

Show `scripts/train.py`: two classifiers trained on the training partition, validation macro F1 selects one, then final test evaluation. Read all numbers directly from `artifacts/metrics.json` or the UI. Explain accuracy versus precision/recall/F1 and read one confusion-matrix row. Show the binary pest model's report as well. Distinguish probability confidence from accuracy and from certainty. Discuss observed class confusions and field-domain limitations honestly.

## P6 — Recommendation API, prevention and demo

Open `/api/recommendations/septoria` to demonstrate a real REST endpoint. Explain the versioned curated KB, sources, and why foreign extension advice is not local pesticide registration. Walk through the inference rules in `cropdetect/knowledge.py`: usable mask AND sufficient confidence AND detector agreement → retrieve guidance; otherwise request review. Demonstrate the upload flow, image stages, 12 features, and rule trace.

## Demo sequence

1. Run `start.ps1` before presenting and open http://127.0.0.1:5000.
2. Confirm the model-ready badge. Read the actual metrics in the evidence section.
3. Select a held-out early-blight sample and click Analyze. Explain candidate versus accepted guidance.
4. Select the spider-mite sample; compare multiclass and separate pest probabilities.
5. Expand the story with before/after/mask images, feature numbers, and the rule trace.
6. If a sample requests expert review, explain why; do not conceal it or change the reported confidence.
7. Show `artifacts/sample_predictions.json` and an incorrect prediction if present.
8. Upload a blank image to show input rejection. Explain that some unrelated photos will still pass the heuristics.
9. End with limitations and a future plan: field dataset, group splits, domain validation, calibrated abstention, true insect bounding-box annotations, local treatment registry integration.

## Likely questions

**Why not deep learning?** The brief requires explicit 12-feature extraction. A classical supervised model makes that pathway transparent and trainable on a laptop. A CNN would be a future comparison, not a substitute for these implemented measurements.

**Where is the expert system?** The ML output supplies uncertain facts; a separate rule engine decides whether to retrieve curated knowledge and produces a trace. It is not just a direct label-to-pesticide string mapping.

**Is this a live pesticide API?** It is a local recommendation REST API backed by a documented curated knowledge base. This option is explicitly permitted in the project brief. It does not query current pesticide availability or registration.

**Can it diagnose any crop?** No. It classifies four supported tomato leaf conditions, with limited input checks. It needs retraining and evaluation for new classes.

**Are mites insects?** No. They are arachnid pests. This pest model detects their damage in leaf images, not individual mite bodies.

**Does high confidence guarantee correctness?** No. Soft class probability can be high on unfamiliar images. The rules add conservative review but are not a substitute for external validation.
