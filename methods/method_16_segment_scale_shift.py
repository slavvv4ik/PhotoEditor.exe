"""Метод 16. Сегментация здания, обнуление фона, масштабирование и смещение."""
import numpy as np
import cv2
from ._utils import ensure_gray_uint8, scale_param, odd_at_least


def process(img):
    img = ensure_gray_uint8(img)
    h, w = img.shape
    # Параметры из оригинала
    blur_ksize = 7
    blur_sigma = 0.0
    seg_min = 10
    seg_max = 148
    border_margin = scale_param(35, img.shape, ref=400.0, min_val=0)
    close_ksize = odd_at_least(scale_param(25, img.shape, ref=400.0, min_val=3))
    open_ksize = odd_at_least(scale_param(15, img.shape, ref=400.0, min_val=3))
    keep_largest = True
    contrast_alpha = 1.3
    scale_factor = 0.55
    offset_x = scale_param(75, img.shape, ref=400.0, min_val=0)
    offset_y = -scale_param(65, img.shape, ref=400.0, min_val=0)

    blurred = cv2.GaussianBlur(img, (odd_at_least(blur_ksize), odd_at_least(blur_ksize)), blur_sigma)
    mask = cv2.inRange(blurred, seg_min, seg_max)

    bm = int(np.clip(border_margin, 0, min(h, w) // 2))
    if bm > 0:
        mask[:bm, :] = 0
        mask[h - bm:, :] = 0
        mask[:, :bm] = 0
        mask[:, w - bm:] = 0

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksize, open_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    if keep_largest:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    segmented = np.zeros_like(img)
    segmented[mask > 0] = img[mask > 0]
    segmented = cv2.convertScaleAbs(segmented, alpha=contrast_alpha, beta=0)

    new_w = max(1, int(w * scale_factor))
    new_h = max(1, int(h * scale_factor))
    resized = cv2.resize(segmented, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((h, w), dtype=np.uint8)
    center_x = (w - new_w) // 2 + offset_x
    center_y = (h - new_h) // 2 + offset_y
    x_start = int(np.clip(center_x, 0, max(0, w - new_w)))
    y_start = int(np.clip(center_y, 0, max(0, h - new_h)))
    canvas[y_start:y_start + new_h, x_start:x_start + new_w] = resized
    return canvas
