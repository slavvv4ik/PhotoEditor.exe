"""Метод 15. Адаптивная бинаризация в симметричный сигнал ±1 с морфологической очисткой."""
import numpy as np
import cv2
from ._utils import ensure_gray_uint8, scale_param, odd_at_least


def process(img):
    img = ensure_gray_uint8(img)
    # Параметры из оригинала
    pre_blur_ksize = 5
    pre_blur_sigma = 0.0
    block_size = 25
    c_const = 4.0
    invert = True
    open_ksize = 3
    close_ksize = 3
    shift_x = scale_param(25, img.shape, ref=400.0, min_val=0)
    shift_y = -scale_param(20, img.shape, ref=400.0, min_val=0)
    level_minus_one = 0
    level_plus_one = 255

    if pre_blur_ksize > 1:
        smoothed = cv2.GaussianBlur(img, (odd_at_least(pre_blur_ksize), odd_at_least(pre_blur_ksize)), pre_blur_sigma)
    else:
        smoothed = img.copy()

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    bs = odd_at_least(block_size)
    if bs < 3:
        bs = 3
    binary = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, bs, c_const)

    cleaned = binary
    if open_ksize > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_ksize), int(open_ksize)))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_open)
    if close_ksize > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_ksize), int(close_ksize)))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    bipolar = np.where(cleaned > 0, 1.0, -1.0).astype(np.float64)
    h, w = img.shape
    if shift_x != 0 or shift_y != 0:
        M = np.float32([[1, 0, float(shift_x)], [0, 1, float(shift_y)]])
        bipolar = cv2.warpAffine(bipolar, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)

    out = np.where(bipolar > 0.0, level_plus_one, level_minus_one).astype(np.uint8)
    return out
