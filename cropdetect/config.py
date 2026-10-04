from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
REVIEW_THRESHOLD = 0.75
CLASSES = {
    'early_blight': 'Tomato___Early_blight',
    'septoria': 'Tomato___Septoria_leaf_spot',
    'spider_mites': 'Tomato___Spider_mites Two-spotted_spider_mite',
    'healthy': 'Tomato___healthy',
}
LABELS = {'early_blight': 'Tomato Early Blight', 'septoria': 'Tomato Septoria Leaf Spot',
          'spider_mites': 'Tomato Spider-mite Damage', 'healthy': 'Tomato Healthy'}
