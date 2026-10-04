"""Measure end-to-end API latency on bundled held-out demo images."""
from __future__ import annotations

import io
import json
import os
import platform
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app


def main() -> None:
    rows = json.loads((ROOT/'artifacts/sample_predictions.json').read_text())
    client = app.test_client()
    first_path = ROOT/rows[0]['path']
    started = time.perf_counter()
    cold_response = client.post('/api/analyze', data={'image': (io.BytesIO(first_path.read_bytes()), first_path.name)})
    cold_start_ms = (time.perf_counter()-started)*1000
    if cold_response.status_code != 200:
        raise RuntimeError(f'Cold benchmark request failed: {cold_response.status_code}')
    timings = []
    iterations = 5
    for _ in range(iterations):
        for row in rows:
            path = ROOT/row['path']
            started = time.perf_counter()
            response = client.post('/api/analyze', data={'image': (io.BytesIO(path.read_bytes()), path.name)})
            elapsed_ms = (time.perf_counter()-started)*1000
            if response.status_code != 200:
                raise RuntimeError(f'Benchmark request failed: {response.status_code}')
            timings.append(elapsed_ms)
    ordered = sorted(timings)
    result = {
        'measurement': 'Flask test client, complete /api/analyze request including decode, preprocessing, feature extraction, both models, reasoning, PNG encoding and JSON serialization',
        'cold_start_ms': cold_start_ms,
        'samples': len(timings),
        'images': len(rows),
        'iterations_per_image': iterations,
        'mean_ms': statistics.fmean(timings),
        'median_ms': statistics.median(timings),
        'p95_ms': ordered[max(0, int(.95*len(ordered))-1)],
        'min_ms': min(timings),
        'max_ms': max(timings),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor() or os.environ.get('PROCESSOR_IDENTIFIER', 'not available'),
        'logical_cpu_count': os.cpu_count(),
        'gpu_used': False,
        'ram_usage': None,
        'vram_usage': None,
        'model_file_bytes': (ROOT/'artifacts/model.joblib').stat().st_size,
        'libraries': {name: version(name) for name in ['Flask', 'scikit-learn', 'scikit-image', 'opencv-python-headless', 'numpy', 'Pillow', 'joblib']},
    }
    path = ROOT/'artifacts/runtime_metrics.json'
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
