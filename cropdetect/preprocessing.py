import cv2
import numpy as np
from PIL import Image, ImageOps


def preprocess(image):
    """Identical train/inference transform; uint8 display plus normalized RGB."""
    image = ImageOps.exif_transpose(image).convert('RGB')
    rgb = np.array(ImageOps.pad(image, (160, 160), color=(128, 128, 128)))
    smooth = cv2.medianBlur(rgb, 3)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV)
    candidate = ((hsv[:, :, 1] > 38) & (hsv[:, :, 2] > 25)).astype('uint8')
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(candidate.shape, np.uint8)
    if contours:
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    enhanced[mask == 0] = 0
    return {'original': rgb, 'enhanced': enhanced, 'normalized': enhanced.astype(np.float32)/255,
            'mask': mask.astype(bool), 'coverage': float(mask.mean())}
