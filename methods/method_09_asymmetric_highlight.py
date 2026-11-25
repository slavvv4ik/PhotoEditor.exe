"""Метод 9. Асимметричное выделение теней и бликов."""
import numpy as np
import cv2
from ._utils import ensure_gray_uint8, scale_param, odd_at_least


def process(img):
    img = ensure_gray_uint8(img)
    # Параметры из оригинала
    median_ksize = 15
    shadow_thresh = 55
    highlight_thresh = 215
    shadow_intensity = 255
    highlight_intensity = 140
    border_crop = scale_param(35, img.shape, ref=400.0, min_val=0)

    smoothed = cv2.medianBlur(img, odd_at_least(median_ksize))
    out = np.zeros_like(img)
    out[smoothed <= shadow_thresh] = shadow_intensity
    out[smoothed >= highlight_thresh] = highlight_intensity

    h, w = out.shape
    bc = int(np.clip(border_crop, 0, min(h, w) // 2))
    if bc > 0:
        out[:bc, :] = 0
        out[h - bc:, :] = 0
        out[:, :bc] = 0
        out[:, w - bc:] = 0
    return out
