# CropDetect: technical case study

Experiment executed locally on 21 September 2026. These are measured results from the included model and artifacts, not figures copied from published papers.

## Objective and scope

Support interpretation of tomato leaf images through a complete image-processing, machine-learning and knowledge-reasoning pipeline. Supported labels are early blight and Septoria leaf spot (fungal diseases), spider-mite damage (an arachnid pest), and healthy leaves. A Python Flask application provides image upload, diagnosis probabilities, a separate pest-damage assessment, visual evidence, treatment guidance and prevention.

## Dataset

Source: [original PlantVillage repository](https://github.com/spMohanty/PlantVillage-Dataset). A deterministic seed-42 sample of 400 images per class was downloaded from a pinned Git revision. Full provenance and SHA-256 values are in `data/manifest.json`.

| Class | Available in source | Used | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Early blight | 1,000 | 400 | 280 | 60 | 60 |
| Healthy | 1,591 | 400 | 280 | 60 | 60 |
| Septoria leaf spot | 1,771 | 400 | 280 | 60 | 60 |
| Spider-mite damage | 1,676 | 400 | 280 | 60 | 60 |
| **Total** | **6,038** | **1,600** | **1,120** | **240** | **240** |

No exact pixel duplicates or unusable masks were found in the selected sample. Stratification preserves class balance. One reflected, brightness-reduced variant is added per training source, resulting in 2,240 training feature rows; validation and test remain at 240 originals each.

## Processing and feature representation

The system corrects EXIF orientation, converts to RGB, resizes with padding to 160×160, applies a median filter, estimates a leaf mask using saturation/value thresholds and a largest-contour rule, and enhances luminance with CLAHE. Background pixels are excluded from feature calculations. Normalization scales color statistics, and the SVM fits a standard scaler on training data only.

The feature vector contains exactly 12 numbers: HSV means (3), hue and saturation standard deviations (2), green/yellow/brown fractions (3), GLCM contrast/homogeneity/energy (3), and contour solidity (1). See the README for equations, numerical thresholds and interpretations. Before/after/mask examples are saved in `artifacts/preprocessing.png`.

## Model selection

| Candidate | Validation macro F1 |
|---|---:|
| Random Forest, 350 trees | 85.85% |
| Standardized RBF SVM, C=10 | **89.16%** |

The RBF SVM was selected using validation data. No hyperparameters were selected using the test set. A separate class-weighted binary RBF SVM was trained to distinguish spider-mite damage from the three other classes.

## Held-out classification results

**Accuracy: 91.67% (220 / 240). Macro precision: 91.79%. Macro recall: 91.67%. Macro F1: 91.66%.**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Early blight | 91.38% | 88.33% | 89.83% | 60 |
| Healthy | 93.44% | 95.00% | 94.21% | 60 |
| Septoria | 87.69% | 95.00% | 91.20% | 60 |
| Spider-mite damage | 94.64% | 88.33% | 91.38% | 60 |

Confusion matrix, rows actual and columns predicted:

| Actual / predicted | Early blight | Healthy | Septoria | Mites |
|---|---:|---:|---:|---:|
| Early blight | 53 | 0 | 5 | 2 |
| Healthy | 2 | 57 | 0 | 1 |
| Septoria | 1 | 2 | 57 | 0 |
| Mites | 2 | 2 | 3 | 53 |

The largest single confusion is early blight predicted as Septoria (5 cases), consistent with the limited specificity of color and texture proxies. This interpretation is a hypothesis about the errors, not a biological validation of the feature set.

## Pest-damage detector

Test accuracy is **95.83%**. For the positive mite class, precision is **90.32%**, recall **93.33%**, and F1 **91.80%**. There are 56 true positives, 4 false negatives, 6 false positives and 174 true negatives. This detector identifies leaf damage patterns; it does not locate mites, count pests, or recognize arbitrary insects.

## Knowledge and reasoning

The knowledge base records disease/pest category, treatment actions, prevention, source URLs, version, and regional scope. A forward-chaining policy derives `supported_candidate` only when mask coverage is between 8% and 95%, the candidate probability reaches 75%, and the binary and multiclass pest decisions agree. Otherwise the result is `needs_review`, with class-specific treatment withheld. Each rule appears in the UI trace.

This demonstrates Module 4 facts, rules, inference, uncertain knowledge and constraints. The threshold is a transparent demonstration policy; it has not been validated for agronomic decisions. Agreement between models is not independent corroboration.

Treatment information comes from a labelled curated knowledge base served by the local recommendation REST API. Sources include [UMN early-blight guidance](https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/early-blight-in-tomato-and-potato), [Maryland Septoria guidance](https://www.extension.umd.edu/resource/septoria-leaf-spot-tomatoes), and [UC IPM spider-mite guidance](https://ipm.ucanr.edu/home-and-landscape/spider-mites/). The API does not claim live product availability or Indian pesticide registration.

## Validation and limitations

Automated tests cover real-image feature extraction and inference, split isolation by pixel checksum, train-only augmentation, rule-based treatment withholding, invalid/oversized uploads, endpoint restrictions, and confusion-matrix consistency. Browser testing exercises sample selection, analysis, displayed confidence, guidance, and the explanatory trace.

The results measure a balanced, controlled-background image-level holdout. They do not establish field accuracy or generalization to unknown diseases. Exact deduplication cannot exclude visually similar or same-plant images. Background segmentation is heuristic. Probability estimates can be overconfident; unknown inputs are not reliably rejected. Further work should prioritize an independent field dataset, plant-level grouping, validated abstention, and an annotated insect dataset if object detection is required.
