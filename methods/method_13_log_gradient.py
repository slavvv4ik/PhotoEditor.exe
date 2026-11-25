"""Метод 13. Контурный препарат: LoG или градиент с мягким порогом."""
import numpy as np
import cv2
from ._utils import ensure_gray_uint8, to_uint8, scale_param, odd_at_least


def process(img):
    img = ensure_gray_uint8(img)
    # Параметры из оригинала
    op_type = "GRADIENT"
    blur_ksize = 5
    blur_sigma = 1.4
    deriv_ksize = 3
    soft_thresh_ratio = 0.15
    soft_steepness = 2.5
    eps = 1e-8
    shift_x = scale_param(25, img.shape, ref=400.0, min_val=0)
    shift_y = -scale_param(20, img.shape, ref=400.0, min_val=0)
    out_max = 255

    img_f = img.astype(np.float64)
    smoothed = cv2.GaussianBlur(img_f, (odd_at_least(blur_ksize), odd_at_least(blur_ksize)), blur_sigma)

    if op_type == "LOG":
        lap = cv2.Laplacian(smoothed, cv2.CV_64F, ksize=odd_at_least(deriv_ksize))
        response = np.abs(lap)
    else:
        dk = odd_at_least(deriv_ksize)
        gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=dk)
        gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=dk)
        response = np.hypot(gx, gy)

    max_resp = float(np.max(response)) + eps
    norm_resp = response / max_resp

    above = np.maximum(0.0, norm_resp - soft_thresh_ratio) / (1.0 - soft_thresh_ratio + eps)
    soft_contour = np.tanh(soft_steepness * above) / np.tanh(soft_steepness + eps)

    h, w = img.shape
    if shift_x != 0 or shift_y != 0:
        M = np.float32([[1, 0, float(shift_x)], [0, 1, float(shift_y)]])
        soft_contour = cv2.warpAffine(soft_contour, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    return to_uint8(soft_contour * out_max)
