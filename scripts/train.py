"""P2-P5: deduplicate, split, preprocess, extract, select, evaluate, persist."""
import csv
import json
import sys
import os
from pathlib import Path
from collections import Counter
import joblib
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, log_loss
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'artifacts/.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cropdetect.config import ROOT, SEED, LABELS, REVIEW_THRESHOLD
from cropdetect.preprocessing import preprocess
from cropdetect.features import extract_features, FEATURE_NAMES


def metrics(y, prediction, labels):
    return {'accuracy': accuracy_score(y, prediction),
            'report': classification_report(y, prediction, labels=labels, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, prediction, labels=labels).tolist(), 'labels': labels}


def probability_metrics(y, prediction, probabilities, classes, pest_probabilities, threshold):
    class_indexes = {label: index for index, label in enumerate(classes)}
    confidence = np.array([probabilities[i, class_indexes[label]] for i, label in enumerate(prediction)])
    correct = prediction == y
    agreement = np.array([(label == 'spider_mites') == (pest >= .5)
                          for label, pest in zip(prediction, pest_probabilities)])
    accepted = (confidence >= threshold) & agreement
    one_hot = np.zeros_like(probabilities)
    for i, label in enumerate(y):
        one_hot[i, class_indexes[label]] = 1
    ece = 0.0
    for lower in np.linspace(0, .9, 10):
        in_bin = (confidence > lower) & (confidence <= lower + .1)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return {
        'temperature_scaling': False,
        'temperature': None,
        'threshold': threshold,
        'above_threshold_count': int((confidence >= threshold).sum()),
        'below_threshold_count': int((confidence < threshold).sum()),
        'accepted_after_pest_agreement_count': int(accepted.sum()),
        'accepted_after_pest_agreement_accuracy': float(correct[accepted].mean()) if accepted.any() else None,
        'multiclass_brier_score': float(np.mean(np.sum((probabilities-one_hot)**2, axis=1))),
        'negative_log_likelihood': float(log_loss(y, probabilities, labels=classes)),
        'ece_10_equal_width_bins': float(ece),
        'note': 'Post-hoc held-out evaluation. SVC probability=True uses internal Platt scaling; no temperature scaling or external field calibration.'
    }


def main():
    out = ROOT/'artifacts'
    out.mkdir(exist_ok=True)
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    unique = {}
    for row in manifest['images']:
        key = row['pixel_sha256']
        if key in unique and unique[key]['label'] != row['label']:
            raise ValueError('Conflicting labels for identical pixels')
        unique.setdefault(key, row)
    rows = list(unique.values())
    train, hold = train_test_split(rows, test_size=.30, stratify=[r['label'] for r in rows], random_state=SEED)
    val, test = train_test_split(hold, test_size=.50, stratify=[r['label'] for r in hold], random_state=SEED)
    partitions = {'train': train, 'validation': val, 'test': test}
    X, y, export, failures = {}, {}, [], []
    for split, items in partitions.items():
        X[split], y[split] = [], []
        for row in items:
            with Image.open(ROOT/row['path']) as image:
                try:
                    vector = extract_features(preprocess(image))
                except ValueError as error:
                    failures.append(dict(row, split=split, error=str(error)))
                    continue
                X[split].append(vector)
                y[split].append(row['label'])
                export.append(dict(path=row['path'], label=row['label'], split=split, augmented=False,
                                   **dict(zip(FEATURE_NAMES, vector.tolist()))))
                if split == 'train':
                    # Augment only after the split; each source stays in its partition.
                    augmented = ImageEnhance.Brightness(image.convert('RGB').transpose(Image.Transpose.FLIP_LEFT_RIGHT)).enhance(.9)
                    aug_vector = extract_features(preprocess(augmented))
                    X[split].append(aug_vector)
                    y[split].append(row['label'])
                    export.append(dict(path=row['path'], label=row['label'], split=split, augmented=True,
                                       **dict(zip(FEATURE_NAMES, aug_vector.tolist()))))
        X[split], y[split] = np.array(X[split]), np.array(y[split])
        print(f'{split}: {len(y[split])} feature rows', flush=True)
    with (out/'features.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(export[0]))
        writer.writeheader()
        writer.writerows(export)
    (out/'splits.json').write_text(json.dumps(partitions, indent=2))
    candidates = {
        'random_forest': RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight='balanced', random_state=SEED, n_jobs=-1),
        'rbf_svm': make_pipeline(StandardScaler(), SVC(C=10, gamma='scale', probability=True, random_state=SEED)),
    }
    scores = {}
    for name, model in candidates.items():
        model.fit(X['train'], y['train'])
        scores[name] = f1_score(y['validation'], model.predict(X['validation']), average='macro')
        print(f'{name} validation macro F1: {scores[name]:.4f}', flush=True)
    selected = max(scores, key=scores.get)
    model = candidates[selected]
    # Binary supervised pest-damage detector, evaluated independently on the same holdout.
    pest = make_pipeline(StandardScaler(), SVC(C=10, probability=True, class_weight='balanced', random_state=SEED))
    pest.fit(X['train'], (y['train'] == 'spider_mites').astype(int))
    labels = sorted(LABELS)
    test_predictions = model.predict(X['test'])
    test_probabilities = model.predict_proba(X['test'])
    pest_test_probabilities = pest.predict_proba(X['test'])[:, 1]
    validation_predictions = model.predict(X['validation'])
    validation_probabilities = model.predict_proba(X['validation'])
    pest_validation_probabilities = pest.predict_proba(X['validation'])[:, 1]
    report = metrics(y['test'], test_predictions, labels)
    report.update({'selected_model': selected, 'validation_macro_f1': scores,
        'validation': metrics(y['validation'], validation_predictions, labels),
        'pest_detector': metrics((y['test']=='spider_mites').astype(int), pest.predict(X['test']), [0, 1]),
        'confidence_policy': {
            'selection_basis': 'Lowest tested threshold reaching at least 95% selective accuracy on validation after pest-model agreement.',
            'validation': probability_metrics(y['validation'], validation_predictions, validation_probabilities,
                                              list(model.classes_), pest_validation_probabilities, REVIEW_THRESHOLD),
            'test': probability_metrics(y['test'], test_predictions, test_probabilities,
                                        list(model.classes_), pest_test_probabilities, REVIEW_THRESHOLD)},
        'source_counts': manifest['available_counts'], 'downloaded': len(manifest['images']),
        'duplicates_removed': len(manifest['images'])-len(rows), 'rejected_images': failures,
        'split_counts': {s: dict(Counter(r['label'] for r in items)) for s, items in partitions.items()},
        'evaluated_rows': {s: len(y[s]) for s in y}, 'seed': SEED,
        'confidence_note': 'Model probability estimate, not a verified diagnostic certainty. No field calibration study.',
        'limitations': ['Controlled-background tomato leaves only.', 'Pixel duplicates removed; visually similar leaves may remain.',
                       'Pest detector recognizes mite damage, not insect species or bounding boxes.',
                       'Image-level holdout; no plant identity metadata or external field evaluation.']})
    joblib.dump({'model': model, 'pest_model': pest, 'features': FEATURE_NAMES, 'threshold': REVIEW_THRESHOLD,
                 'selected_model': selected}, out/'model.joblib')
    (out/'metrics.json').write_text(json.dumps(report, indent=2))
    fig, ax = plt.subplots(figsize=(7, 6))
    matrix = np.array(report['confusion_matrix'])
    ax.imshow(matrix, cmap='Greens')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i, j], ha='center', va='center')
    ax.set(xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels,
           xlabel='Predicted', ylabel='Actual', title='Untouched test set • confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    fig.tight_layout(); fig.savefig(out/'confusion_matrix.png', dpi=150); plt.close(fig)
    # Pick one representative, correctly classified held-out image per class for
    # the live demo. This affects only the sample buttons, never evaluation.
    pest_probabilities = pest_test_probabilities
    demo_rows = []
    for label in labels:
        candidates = []
        for row, vector, predicted, probabilities, pest_probability in zip(
                test, X['test'], test_predictions, test_probabilities, pest_probabilities):
            class_probability = float(probabilities[list(model.classes_).index(predicted)])
            detectors_agree = ((predicted == 'spider_mites') == (pest_probability >= .5))
            if row['label'] == label and predicted == label and detectors_agree:
                candidates.append((class_probability, row, vector, float(pest_probability)))
        if not candidates:
            raise RuntimeError(f'No correctly classified, detector-agreeing test sample for {label}')
        demo_rows.append(max(candidates, key=lambda item: item[0]))

    fig, axes = plt.subplots(4, 3, figsize=(9, 11))
    samples = []
    for i, (label, selected) in enumerate(zip(labels, demo_rows)):
        confidence, row, vector, pest_probability = selected
        with Image.open(ROOT/row['path']) as image:
            processed = preprocess(image)
        for ax, key in zip(axes[i], ['original', 'enhanced', 'mask']):
            ax.imshow(processed[key], cmap='gray'); ax.axis('off'); ax.set_title(label+' / '+key, fontsize=9)
        samples.append({'path': row['path'], 'actual': label, 'predicted': label,
                        'confidence': confidence, 'pest_probability': pest_probability,
                        'selection': 'Highest-confidence correct test prediction with multiclass/binary agreement; excluded from model selection.'})
    fig.tight_layout(); fig.savefig(out/'preprocessing.png', dpi=130); plt.close(fig)
    (out/'sample_predictions.json').write_text(json.dumps(samples, indent=2))
    print(json.dumps({'test_accuracy': report['accuracy'], 'test_macro_f1': report['report']['macro avg']['f1-score']}), flush=True)


if __name__ == '__main__':
    main()
