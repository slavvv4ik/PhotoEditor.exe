"""Метод 3. Морфология: морфологическое открытие + растяжение."""
import numpy as np
import cv2
from ._utils import ensure_gray_uint8, to_uint8, odd_at_least


def process(img):
    img = ensure_gray_uint8(img)
    size = 7  # из оригинала
    k = odd_at_least(size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    result = opened.astype(np.float64)
    mn, mx = float(result.min()), float(result.max())
    result = (result - mn) / (mx - mn + 1e-12) * 255.0
    return to_uint8(result)
