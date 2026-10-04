import cv2
import numpy as np
from skimage.feature import graycoprops

FEATURE_NAMES = ['hue_mean', 'saturation_mean', 'value_mean', 'hue_std', 'saturation_std',
                 'green_fraction', 'yellow_fraction', 'brown_fraction', 'glcm_contrast',
                 'glcm_homogeneity', 'glcm_energy', 'leaf_solidity']


def extract_features(processed):
    """Exactly 12 numeric features; texture pairs exclude the masked background."""
    rgb, mask = processed['enhanced'], processed['mask']
    if mask.sum() < 64:
        raise ValueError('No usable leaf region. Use a clear single leaf on a plain background.')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(float)
    h, s, v = (hsv[:, :, i][mask] for i in range(3))
    values = [h.mean()/179, s.mean()/255, v.mean()/255, h.std()/179, s.std()/255,
              np.mean((h >= 35) & (h <= 90)), np.mean((h >= 18) & (h < 35)),
              np.mean((h < 25) & (s > 65) & (v < 180))]
    gray = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)//8).astype(int)
    matrices = []
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        y0, y1 = max(0, -dy), min(160, 160-dy)
        x0, x1 = max(0, -dx), min(160, 160-dx)
        a, b = gray[y0:y1, x0:x1], gray[y0+dy:y1+dy, x0+dx:x1+dx]
        valid = mask[y0:y1, x0:x1] & mask[y0+dy:y1+dy, x0+dx:x1+dx]
        counts = np.bincount(a[valid]*32+b[valid], minlength=1024).reshape(32, 32).astype(float)
        counts += counts.T.copy()
        matrices.append(counts/max(counts.sum(), 1))
    glcm = np.stack(matrices, axis=-1)[:, :, None, :]
    values += [graycoprops(glcm, p).mean() for p in ['contrast', 'homogeneity', 'energy']]
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    values += [cv2.contourArea(contour)/max(cv2.contourArea(cv2.convexHull(contour)), 1)]
    result = np.array(values, dtype=float)
    if result.shape != (12,) or not np.isfinite(result).all():
        raise ValueError('Invalid feature vector')
    return result
