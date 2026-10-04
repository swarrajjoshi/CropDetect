"""Download a reproducible, balanced subset from the original public repository."""
import argparse
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cropdetect.config import ROOT, CLASSES, SEED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=400)
    args = parser.parse_args()
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1)))
    repo = 'https://api.github.com/repos/spMohanty/PlantVillage-Dataset'
    index_path = ROOT/'data/source_index.json'
    index_path.parent.mkdir(exist_ok=True)
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        response = session.get(repo+'/git/trees/master?recursive=1', timeout=90)
        response.raise_for_status()
        index = response.json()
        index_path.write_text(json.dumps(index))
    if index.get('truncated'):
        color = next(t for t in index['tree'] if t['path'] == 'raw/color')
        response = session.get(repo+'/git/trees/'+color['sha'], timeout=90)
        response.raise_for_status()
        subtree = response.json()
        entries = []
        for folder in CLASSES.values():
            folder_entry = next(t for t in subtree['tree'] if t['path'] == folder)
            response = session.get(repo+'/git/trees/'+folder_entry['sha'], timeout=90)
            response.raise_for_status()
            listing = response.json()
            if listing.get('truncated'):
                raise RuntimeError('Class listing truncated')
            entries.extend(dict(t, path='raw/color/'+folder+'/'+t['path']) for t in listing['tree'])
        index['tree'] = entries
        index['truncated'] = False
        index_path.write_text(json.dumps(index))
    tasks, counts = [], {}
    for label, folder in CLASSES.items():
        candidates = sorted(t['path'] for t in index['tree'] if t['path'].startswith('raw/color/'+folder+'/') and t['path'].lower().endswith(('.jpg', '.png', '.jpeg')))
        counts[label] = len(candidates)
        if len(candidates) < args.per_class:
            raise ValueError(f'{folder}: only {len(candidates)} available')
        tasks += [(label, p) for p in random.Random(SEED).sample(candidates, args.per_class)]
    def download(task):
        label, source = task
        destination = ROOT/'data/raw'/label/Path(source).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/'+index['sha']+'/'+quote(source)
        if not destination.exists():
            response = session.get(url, timeout=60)
            response.raise_for_status()
            destination.write_bytes(response.content)
        with Image.open(destination) as image:
            pixels = image.convert('RGB').tobytes()
        return {'label': label, 'path': destination.relative_to(ROOT).as_posix(), 'url': url,
                'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                'pixel_sha256': hashlib.sha256(pixels).hexdigest()}
    rows = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i, row in enumerate(pool.map(download, tasks), 1):
            rows.append(row)
            if i % 100 == 0:
                print(f'Downloaded and verified {i}/{len(tasks)}', flush=True)
    manifest = {'source': 'https://github.com/spMohanty/PlantVillage-Dataset', 'revision': index['sha'],
                'seed': SEED, 'available_counts': counts, 'requested_per_class': args.per_class, 'images': rows}
    (ROOT/'data/manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Saved data/manifest.json', flush=True)


if __name__ == '__main__':
    main()
