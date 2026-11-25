"""Метод 5. Согласованный фильтр: отклик 'квадрат минус кольцо' как мягкая маска."""
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


def _local_mean(f, k):
    h, w = f.shape
    kernel = np.ones((k, k), dtype=np.float64) / float(k * k)
    size = (h + k - 1, w + k - 1)
    K = np.fft.fft2(kernel, s=size)
    num = np.fft.ifft2(np.fft.fft2(f, s=size) * K).real
    den = np.fft.ifft2(np.fft.fft2(np.ones_like(f), s=size) * K).real
    mean = num / np.maximum(den, 1e-9)
    y0 = k // 2
    x0 = k // 2
    return mean[y0:y0 + h, x0:x0 + w]


def process(img):
    img = ensure_gray_uint8(img)
    # Параметры из оригинала
    window_smooth = 5
    gate_power = 2.0
    mask_percentile = 99.5
    ring_multiplier = 2

    smoothed = cv2.medianBlur(img, odd_at_least(window_smooth)).astype(np.float64)
    thr = _otsu_threshold(smoothed)
    area = int((smoothed > thr).sum())
    h, w = img.shape
    side = max(3, int(round(np.sqrt(max(area, 1)))) | 1)
    side = min(side, min(h, w) - 2) if min(h, w) > 4 else 3
    if side % 2 == 0:
        side += 1
    neighborhood = ring_multiplier * side + 1
    neighborhood = min(neighborhood, min(h, w) - (1 - min(h, w) % 2))
    if neighborhood <= side:
        neighborhood = side + 2
    if neighborhood % 2 == 0:
        neighborhood += 1
    neighborhood = min(neighborhood, min(h, w) if min(h, w) % 2 == 1 else min(h, w) - 1)
    if neighborhood <= side:
        # fallback: simple stretch
        arr = img.astype(np.float64)
        mn, mx = float(arr.min()), float(arr.max())
        return to_uint8((arr - mn) / (mx - mn + 1e-12) * 255.0)

    arr = img.astype(np.float64)
    inner = _local_mean(arr, side)
    outer = _local_mean(arr, neighborhood)
    s_in = float(side * side)
    s_out = float(neighborhood * neighborhood)
    ring_mean = (outer * s_out - inner * s_in) / max(s_out - s_in, 1e-9)
    response = inner - ring_mean
    ref = float(np.percentile(response, mask_percentile))
    weight = np.clip(response / (ref + 1e-12), 0.0, 1.0)
    weight = np.power(weight, gate_power)
    result = arr * weight
    mn, mx = float(result.min()), float(result.max())
    result = (result - mn) / (mx - mn + 1e-12) * 255.0
    return to_uint8(result)
