"""Метод 7. Структурный каркас: bilateral + Canny + дилатация + сдвиг."""
import numpy as np
import cv2
from ._utils import ensure_gray_uint8, scale_param


def process(img):
    img = ensure_gray_uint8(img)
    # Параметры из оригинала
    bilateral_d = 10
    sigma_color = 75
    sigma_space = 75
    canny_low = 60
    canny_high = 160
    dilate_size = 2
    shift_x = scale_param(25, img.shape, ref=400.0, min_val=0)
    shift_y = -scale_param(20, img.shape, ref=400.0, min_val=0)
    border_val = 0

    filtered = cv2.bilateralFilter(img, bilateral_d, sigma_color, sigma_space)
    edges = cv2.Canny(filtered, canny_low, canny_high)
    k = max(1, int(dilate_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    thick = cv2.dilate(edges, kernel)
    h, w = img.shape
    M = np.float32([[1, 0, float(shift_x)], [0, 1, float(shift_y)]])
    shifted = cv2.warpAffine(thick, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    return shifted
