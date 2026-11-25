"""Метод 1. Порог Отсу: медиана -> порог Отсу -> мягкое подавление фона -> медиана."""
import numpy as np
import cv2
from ._utils import ensure_gray_uint8, to_uint8, odd_at_least


def _otsu_threshold(img_f64):
    hist, _ = np.histogram(img_f64.ravel(), bins=256, range=(0, 256))
    prob = hist.astype(np.float64)
    total = prob.sum()
    if total <= 0:
        return 0.0
    prob /= total
    levels = np.arange(256, dtype=np.float64)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * levels)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega) + 1e-12
    sigma_b = (mu_total * omega - mu) ** 2 / denom
    return float(np.argmax(sigma_b))


def process(img):
    img = ensure_gray_uint8(img)
    # Параметры из оригинала
    window_smooth = 5
    window_cleanup = 3
    percentile_top = 99.5

    smoothed = cv2.medianBlur(img, odd_at_least(window_smooth))
    smoothed_f = smoothed.astype(np.float64)
    thr = _otsu_threshold(smoothed_f)
    top = float(np.percentile(smoothed_f, percentile_top))
    gain = 255.0 / max(top - thr, 1.0)
    result = np.clip((smoothed_f - thr) * gain, 0.0, 255.0)
    result_u8 = to_uint8(result)
    result_u8 = cv2.medianBlur(result_u8, odd_at_least(window_cleanup))
    return result_u8
