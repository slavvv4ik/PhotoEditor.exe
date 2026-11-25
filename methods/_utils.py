import numpy as np
import cv2


def ensure_gray_uint8(img):
    """Accept uint8/float grayscale or BGR/BGRA; return uint8 grayscale."""
    if img is None:
        raise ValueError("Empty image")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def to_float(img_uint8):
    return img_uint8.astype(np.float64)


def to_uint8(img_float):
    return np.round(np.clip(img_float, 0, 255)).astype(np.uint8)


def scale_param(value, img_shape, ref=400.0, min_val=1):
    """Scale a pixel parameter relative to image size (min dimension / ref)."""
    h, w = img_shape[:2]
    s = min(h, w) / float(ref)
    scaled = int(round(value * s))
    return max(min_val, scaled)


def odd_at_least(value, minimum=3):
    v = int(value)
    if v < minimum:
        v = minimum
    if v % 2 == 0:
        v += 1
    return v


def contrast_score(img_uint8):
    """Higher = more contrast. Standard deviation of luminance."""
    return float(np.std(img_uint8.astype(np.float64)))
