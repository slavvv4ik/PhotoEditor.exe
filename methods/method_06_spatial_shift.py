"""Метод 6. Пространственный сдвиг: подавление фона, масштабирование, смещение."""
import numpy as np
import cv2
from ._utils import ensure_gray_uint8, to_uint8, scale_param, odd_at_least


def process(img):
    img = ensure_gray_uint8(img)
    # Параметры из оригинала (масштабируются под размер изображения)
    blur_ksize = 5
    blur_sigma = 0
    bg_low = 95
    bg_high = 175
    scale = 0.55
    shift_x = scale_param(180, img.shape)
    shift_y = scale_param(20, img.shape)
    contrast_alpha = 1.4
    contrast_beta = 0

    blurred = cv2.GaussianBlur(img, (odd_at_least(blur_ksize), odd_at_least(blur_ksize)), blur_sigma)
    mask = (blurred < bg_low) | (blurred > bg_high)
    cleaned = np.zeros_like(img)
    cleaned[mask] = img[mask]
    cleaned = cv2.convertScaleAbs(cleaned, alpha=contrast_alpha, beta=contrast_beta)

    h, w = img.shape
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(cleaned, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w), dtype=np.uint8)
    x_start = int(np.clip(shift_x, 0, max(0, w - new_w)))
    y_start = int(np.clip(shift_y, 0, max(0, h - new_h)))
    canvas[y_start:y_start + new_h, x_start:x_start + new_w] = resized
    return canvas
