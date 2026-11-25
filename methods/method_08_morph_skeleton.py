"""Метод 8. Морфологическая скелетизация силуэта."""
import numpy as np
import cv2
from ._utils import ensure_gray_uint8, scale_param, odd_at_least


def process(img):
    img = ensure_gray_uint8(img)
    # Параметры из оригинала
    blur_ksize = 15
    blur_sigma = 0
    roof_min = 85
    roof_max = 145
    close_ksize = 21
    open_ksize = 15
    skel_elem_size = 3
    skel_thickness = 3
    shift_x = scale_param(35, img.shape, ref=400.0, min_val=0)
    shift_y = scale_param(30, img.shape, ref=400.0, min_val=0)
    border_val = 0

    blurred = cv2.GaussianBlur(img, (odd_at_least(blur_ksize), odd_at_least(blur_ksize)), blur_sigma)
    mask = cv2.inRange(blurred, roof_min, roof_max)

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (odd_at_least(close_ksize), odd_at_least(close_ksize)))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (odd_at_least(open_ksize), odd_at_least(open_ksize)))
    solid = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, k_open)

    skeleton = np.zeros(solid.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (odd_at_least(skel_elem_size), odd_at_least(skel_elem_size)))
    temp_img = solid.copy()
    # Safety limit to avoid pathological loops
    max_iter = max(img.shape) * 2
    for _ in range(max_iter):
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        temp_img = eroded.copy()
        if cv2.countNonZero(temp_img) == 0:
            break

    if skel_thickness > 1:
        k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (int(skel_thickness), int(skel_thickness)))
        skeleton = cv2.dilate(skeleton, k_dil)

    h, w = img.shape
    M = np.float32([[1, 0, float(shift_x)], [0, 1, float(shift_y)]])
    shifted = cv2.warpAffine(skeleton, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    return shifted
