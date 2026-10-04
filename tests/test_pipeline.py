import io
import json
import numpy as np
import pytest
from PIL import Image
from app import app
from cropdetect.config import ROOT
from cropdetect.features import FEATURE_NAMES, extract_features
from cropdetect.preprocessing import preprocess
from cropdetect.knowledge import reason, KB


def test_uncertain_or_conflicting_predictions_withhold_treatment():
    for confidence, pest, coverage in [(.4, .1, .5), (.9, .9, .5), (.9, .1, .01)]:
        result = reason('early_blight', confidence, pest, coverage)
        assert result['status'] == 'needs_review'
        assert result['recommendation']['provider'] == 'Uncertainty policy'
    assert reason('early_blight', .9, .1, .5)['status'] == 'supported_candidate'
    assert reason('spider_mites', .9, .9, .5)['status'] == 'supported_candidate'


def test_real_sample_features_and_prediction():
    samples = json.loads((ROOT/'artifacts/sample_predictions.json').read_text())
    client = app.test_client()
    for row in samples:
        path = ROOT/row['path']
        with Image.open(path) as image:
            features = extract_features(preprocess(image))
        assert len(FEATURE_NAMES) == len(features) == 12
        assert np.isfinite(features).all()
        response = client.post('/api/analyze', data={'image': (io.BytesIO(path.read_bytes()), 'leaf.jpg')})
        assert response.status_code == 200
        data = response.get_json()
        assert data['label'] == row['predicted']
        assert abs(sum(data['probabilities'].values())-1) < 1e-6
        assert len(data['features']) == 12
        assert len(data['inference']['trace']) == 4


def test_split_integrity():
    splits = json.loads((ROOT/'artifacts/splits.json').read_text())
    groups = [{r['pixel_sha256'] for r in rows} for rows in splits.values()]
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            assert groups[i].isdisjoint(groups[j])
    with (ROOT/'artifacts/features.csv').open() as handle:
        import csv
        for row in csv.DictReader(handle):
            assert row['augmented'] == 'False' or row['split'] == 'train'


def test_bad_inputs_and_routes():
    client = app.test_client()
    assert client.post('/api/analyze').status_code == 400
    assert client.post('/api/analyze', data={'image': (io.BytesIO(b'not an image'), 'bad.jpg')}).status_code == 400
    stream = io.BytesIO(); Image.new('RGB', (160, 160), 'gray').save(stream, format='PNG'); stream.seek(0)
    assert client.post('/api/analyze', data={'image': (stream, 'blank.png')}).status_code == 400
    assert client.post('/api/analyze', data={'image': (io.BytesIO(b'x'*(9*1024*1024)), 'large.png')}).status_code == 413
    assert client.get('/api/recommendations/unknown').status_code == 404
    assert client.get('/reports/model.joblib').status_code == 404
    assert client.get('/sample/999').status_code == 404
    for label in KB:
        response = client.get('/api/recommendations/'+label)
        assert response.status_code == 200
        assert response.get_json()['sources']


def test_public_pages_health_and_deployable_samples():
    client = app.test_client()
    assert client.get('/').status_code == 200
    health = client.get('/api/health')
    assert health.status_code == 200
    assert health.get_json()['status'] == 'ok'
    samples = client.get('/api/samples').get_json()
    assert len(samples) == 4
    for index, row in enumerate(samples):
        response = client.get(f'/sample/{index}')
        assert response.status_code == 200
        assert response.mimetype == 'image/jpeg'
        assert row['actual'] in KB


def test_metrics_consistent():
    m = json.loads((ROOT/'artifacts/metrics.json').read_text())
    matrix = np.array(m['confusion_matrix'])
    assert matrix.sum() == m['evaluated_rows']['test']
    assert m['accuracy'] == pytest.approx(matrix.trace()/matrix.sum())
