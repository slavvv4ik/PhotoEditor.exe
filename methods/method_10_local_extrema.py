"""Метод 10. Локальные экстремумы (Top-Hat / Black-Hat)."""
import numpy as np
import cv2
from ._utils import ensure_gray_uint8, scale_param, odd_at_least


def process(img):
    img = ensure_gray_uint8(img)
    # Параметры из оригинала
    tophat_ksize = odd_at_least(scale_param(19, img.shape, ref=400.0, min_val=3))
    thresh_val = 75
    max_val = 255
    dilate_ksize = 3
    border_margin = scale_param(45, img.shape, ref=400.0, min_val=0)
    shift_x = scale_param(20, img.shape, ref=400.0, min_val=0)
    shift_y = -scale_param(25, img.shape, ref=400.0, min_val=0)
    border_val = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_ksize, tophat_ksize))
    white_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    black_tophat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    combined = cv2.add(white_tophat, black_tophat)

    h, w = combined.shape
    bm = int(np.clip(border_margin, 0, min(h, w) // 2))
    if bm > 0:
        combined[:bm, :] = 0
        combined[h - bm:, :] = 0
        combined[:, :bm] = 0
        combined[:, w - bm:] = 0

    _, keypoints = cv2.threshold(combined, thresh_val, max_val, cv2.THRESH_BINARY)
    if dilate_ksize > 1:
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dilate_ksize), int(dilate_ksize)))
        keypoints = cv2.dilate(keypoints, k_dil)

    M = np.float32([[1, 0, float(shift_x)], [0, 1, float(shift_y)]])
    shifted = cv2.warpAffine(keypoints, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    return shifted
