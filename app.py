import base64
import io
import json
import os
import warnings
import joblib
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from cropdetect.config import ROOT, LABELS
from cropdetect.preprocessing import preprocess
from cropdetect.features import extract_features, FEATURE_NAMES
from cropdetect.knowledge import recommendation, reason

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8*1024*1024
Image.MAX_IMAGE_PIXELS = 20_000_000
bundle = None


def get_model():
    global bundle
    if bundle is None:
        path = ROOT/'artifacts/model.joblib'
        if not path.exists():
            raise FileNotFoundError('Model unavailable. Run scripts/download_dataset.py then scripts/train.py.')
        bundle = joblib.load(path)
    return bundle


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/health')
def health():
    model_ready = (ROOT/'artifacts/model.joblib').exists()
    return jsonify(status='ok' if model_ready else 'degraded', model_ready=model_ready, classes=LABELS)


@app.get('/api/metrics')
def metrics():
    path = ROOT/'artifacts/metrics.json'
    return (jsonify(json.loads(path.read_text())) if path.exists() else (jsonify(error='Train the model first.'), 503))


@app.get('/api/recommendations/<label>')
def recommendations(label):
    if label not in LABELS:
        return jsonify(error='Unknown class'), 404
    return jsonify(recommendation(label))


@app.get('/reports/<name>')
def reports(name):
    if name not in ['confusion_matrix.png', 'preprocessing.png', 'metrics.json', 'sample_predictions.json']:
        return jsonify(error='Not found'), 404
    return send_from_directory(ROOT/'artifacts', name)


@app.get('/api/samples')
def samples():
    path = ROOT/'artifacts/sample_predictions.json'
    return jsonify(json.loads(path.read_text()) if path.exists() else [])


@app.get('/sample/<int:index>')
def sample(index):
    rows = json.loads((ROOT/'artifacts/sample_predictions.json').read_text())
    if not 0 <= index < len(rows):
        return jsonify(error='Not found'), 404
    deployed_sample = ROOT/'static'/'samples'/f'{index}.jpg'
    if deployed_sample.exists():
        return send_from_directory(deployed_sample.parent, deployed_sample.name)
    path = ROOT/rows[index]['path']
    if not path.exists():
        return jsonify(error='Sample image unavailable'), 404
    return send_from_directory(path.parent, path.name)


@app.post('/api/analyze')
def analyze():
    if 'image' not in request.files:
        return jsonify(error='Choose an image first.'), 400
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(request.files['image'].stream) as image:
                if image.format not in ['JPEG', 'PNG', 'WEBP']:
                    raise ValueError('Use a JPEG, PNG, or WebP image.')
                processed = preprocess(image)
        vector = extract_features(processed)
        model = get_model()
        probabilities = model['model'].predict_proba(vector[None, :])[0]
        labels = model['model'].classes_
        # Match the classifier decision used in held-out evaluation. SVC's
        # calibrated probability argmax can occasionally differ from predict().
        decision = model['model'].predict(vector[None, :])[0]
        best = list(labels).index(decision)
        label = str(labels[best])
        pest_probability = float(model['pest_model'].predict_proba(vector[None, :])[0, 1])
        inference = reason(label, float(probabilities[best]), pest_probability, processed['coverage'], model['threshold'])
        return jsonify(label=label, prediction=LABELS[label], confidence=float(probabilities[best]),
            probabilities={LABELS[str(k)]: float(v) for k, v in zip(labels, probabilities)},
            pest_probability=pest_probability, pest_scope='Binary ML detection of spider-mite leaf damage; no insect localization.',
            features=dict(zip(FEATURE_NAMES, vector.tolist())), inference=inference,
            images={'before': png(processed['original']), 'after': png(processed['enhanced']),
                    'mask': png(processed['mask'].astype('uint8')*255)})
    except FileNotFoundError as error:
        return jsonify(error=str(error)), 503
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        return jsonify(error=str(error) or 'Invalid image'), 400


@app.errorhandler(413)
def too_large(error):
    return jsonify(error='Image must be smaller than 8 MB.'), 413


if __name__ == '__main__':
    app.run(host=os.getenv('HOST', '127.0.0.1'), port=int(os.getenv('PORT', '5000')), debug=False)
