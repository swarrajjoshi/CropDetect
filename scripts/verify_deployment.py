"""Deployment smoke test using real held-out images and public HTTP routes."""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app
from cropdetect.config import LABELS, ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=5)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts'/'verification_results.json')
    args = parser.parse_args()

    test_rows = json.loads((ROOT/'artifacts/splits.json').read_text())['test']
    selected: dict[str, list[dict]] = defaultdict(list)
    for row in test_rows:
        if len(selected[row['label']]) < args.per_class:
            selected[row['label']].append(row)

    checks = []
    predictions = []
    client = app.test_client()

    def check(name: str, passed: bool, detail: str = '') -> None:
        checks.append({'name': name, 'passed': bool(passed), 'detail': detail})

    root_response = client.get('/')
    check('homepage', root_response.status_code == 200, str(root_response.status_code))
    health = client.get('/api/health')
    check('health', health.status_code == 200 and health.get_json().get('model_ready'), str(health.get_json()))
    check('metrics', client.get('/api/metrics').status_code == 200)

    for label in LABELS:
        response = client.get(f'/api/recommendations/{label}')
        body = response.get_json()
        check(f'recommendation:{label}', response.status_code == 200 and bool(body.get('sources')))

    samples = client.get('/api/samples').get_json()
    check('sample-metadata', len(samples) == 4, f'{len(samples)} rows')
    for index in range(len(samples)):
        response = client.get(f'/sample/{index}')
        check(f'sample-image:{index}', response.status_code == 200 and response.mimetype == 'image/jpeg')

    for label, rows in selected.items():
        for row in rows:
            path = ROOT/row['path']
            response = client.post('/api/analyze', data={'image': (io.BytesIO(path.read_bytes()), path.name)})
            body = response.get_json()
            passed = (
                response.status_code == 200
                and len(body.get('features', {})) == 12
                and abs(sum(body.get('probabilities', {}).values()) - 1.0) < 1e-6
                and len(body.get('inference', {}).get('trace', [])) == 4
            )
            check(f'analyze:{label}:{path.name}', passed, str(response.status_code))
            if response.status_code == 200:
                predictions.append({'actual': label, 'predicted': body['label'], 'confidence': body['confidence']})

    source = ROOT/next(iter(selected.values()))[0]['path']
    with Image.open(source) as image:
        for image_format, extension in [('PNG', 'png'), ('WEBP', 'webp')]:
            stream = io.BytesIO()
            image.save(stream, format=image_format)
            stream.seek(0)
            response = client.post('/api/analyze', data={'image': (stream, f'leaf.{extension}')})
            check(f'accepted-format:{extension}', response.status_code == 200, str(response.status_code))

    invalid_cases = {
        'missing-file': client.post('/api/analyze'),
        'corrupt-jpeg': client.post('/api/analyze', data={'image': (io.BytesIO(b'not an image'), 'bad.jpg')}),
    }
    gif = io.BytesIO()
    Image.new('RGB', (32, 32), 'green').save(gif, format='GIF')
    gif.seek(0)
    invalid_cases['unsupported-gif'] = client.post('/api/analyze', data={'image': (gif, 'leaf.gif')})
    blank = io.BytesIO()
    Image.new('RGB', (160, 160), 'gray').save(blank, format='PNG')
    blank.seek(0)
    invalid_cases['blank-image'] = client.post('/api/analyze', data={'image': (blank, 'blank.png')})
    invalid_cases['oversized-upload'] = client.post(
        '/api/analyze', data={'image': (io.BytesIO(b'x'*(9*1024*1024)), 'large.png')}
    )
    expected = {'missing-file': 400, 'corrupt-jpeg': 400, 'unsupported-gif': 400, 'blank-image': 400, 'oversized-upload': 413}
    for name, response in invalid_cases.items():
        check(f'reject:{name}', response.status_code == expected[name], str(response.status_code))

    correct = sum(row['actual'] == row['predicted'] for row in predictions)
    result = {
        'passed': all(item['passed'] for item in checks),
        'checks_total': len(checks),
        'checks_passed': sum(item['passed'] for item in checks),
        'real_images_tested': len(predictions),
        'real_image_accuracy_for_smoke_subset': correct/len(predictions) if predictions else None,
        'prediction_counts': Counter(row['predicted'] for row in predictions),
        'predictions': predictions,
        'checks': checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=dict))
    print(json.dumps({key: value for key, value in result.items() if key not in {'checks', 'predictions'}}, indent=2, default=dict))
    print(f'Full results: {args.output}')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
