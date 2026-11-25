"""Метод 14. Гомоморфная локальная нормализация."""
import numpy as np
import cv2
from ._utils import ensure_gray_uint8, to_uint8, scale_param, odd_at_least


def process(img):
    img = ensure_gray_uint8(img)
    h, w = img.shape
    # Параметры из оригинала; крупные ядра масштабируем под размер
    log_eps = 1.0
    bg_ksize = odd_at_least(scale_param(51, img.shape, ref=400.0, min_val=15))
    bg_sigma = 20.0 * min(h, w) / 400.0
    std_ksize = odd_at_least(scale_param(21, img.shape, ref=400.0, min_val=5))
    std_sigma = 7.0 * min(h, w) / 400.0
    std_reg = 0.05
    clip_limit = 3.0
    out_min = 0
    out_max = 255
    shift_x = scale_param(20, img.shape, ref=400.0, min_val=0)
    shift_y = -scale_param(15, img.shape, ref=400.0, min_val=0)

    # kernel sizes must not exceed image dimensions in a pathological way; OpenCV handles it, but clamp
    bg_ksize = min(bg_ksize, (min(h, w) // 2) * 2 + 1)
    std_ksize = min(std_ksize, (min(h, w) // 2) * 2 + 1)
    bg_ksize = odd_at_least(max(3, bg_ksize))
    std_ksize = odd_at_least(max(3, std_ksize))

    img_f = img.astype(np.float64)
    log_img = np.log(img_f + log_eps)

    log_bg = cv2.GaussianBlur(log_img, (bg_ksize, bg_ksize), bg_sigma)
    high_pass = log_img - log_bg

    local_mean = cv2.GaussianBlur(high_pass, (std_ksize, std_ksize), std_sigma)
    local_sq_mean = cv2.GaussianBlur(high_pass ** 2, (std_ksize, std_ksize), std_sigma)
    local_var = np.maximum(0.0, local_sq_mean - local_mean ** 2)
    local_std = np.sqrt(local_var)

    normalized = (high_pass - local_mean) / (local_std + std_reg)

    if shift_x != 0 or shift_y != 0:
        M = np.float32([[1, 0, float(shift_x)], [0, 1, float(shift_y)]])
        normalized = cv2.warpAffine(normalized, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    clipped = np.clip(normalized, -clip_limit, clip_limit) / clip_limit
    scaled = ((clipped + 1.0) * 0.5) * (out_max - out_min) + out_min
    return to_uint8(scaled)
