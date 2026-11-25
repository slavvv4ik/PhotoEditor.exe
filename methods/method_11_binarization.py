"""Метод 11. Бинаризация (OTSU/MANUAL/ADAPTIVE)."""
import cv2
from ._utils import ensure_gray_uint8, odd_at_least


def process(img):
    img = ensure_gray_uint8(img)
    # Параметры из оригинала
    blur_ksize = 13
    blur_sigma = 0
    method = "OTSU"
    manual_thresh = 127
    max_val = 255
    block_size = 15
    c_const = 4
    invert = False

    if blur_ksize > 1:
        img = cv2.GaussianBlur(img, (odd_at_least(blur_ksize), odd_at_least(blur_ksize)), blur_sigma)

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if method == "OTSU":
        _, binary = cv2.threshold(img, manual_thresh, max_val, thresh_type + cv2.THRESH_OTSU)
    elif method == "ADAPTIVE":
        bs = odd_at_least(block_size)
        if bs < 3:
            bs = 3
        binary = cv2.adaptiveThreshold(img, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, bs, c_const)
    else:
        _, binary = cv2.threshold(img, manual_thresh, max_val, thresh_type)
    return binary
