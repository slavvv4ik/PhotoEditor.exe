"""Метод 12. Вычитание постоянной составляющей + окно Тьюки/Ханна + нормировка."""
import numpy as np
import cv2
from ._utils import ensure_gray_uint8, to_uint8, scale_param


def _create_1d_window(length, win_type, tukey_alpha):
    x = np.linspace(0.0, 1.0, length)
    if win_type == "HANN" or tukey_alpha >= 1.0:
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * x))
    elif tukey_alpha <= 0.0:
        return np.ones(length, dtype=np.float64)
    else:
        win = np.ones(length, dtype=np.float64)
        half_alpha = tukey_alpha / 2.0
        mask1 = x < half_alpha
        win[mask1] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[mask1] / tukey_alpha - 1.0)))
        mask2 = x >= (1.0 - half_alpha)
        win[mask2] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[mask2] / tukey_alpha - 2.0 / tukey_alpha + 1.0)))
        return win


def process(img):
    img = ensure_gray_uint8(img)
    # Параметры из оригинала
    win_type = "TUKEY"
    tukey_alpha = 0.35
    dc_mode = "WINDOWED_MEAN"
    clip_sigma = 3.0
    eps = 1e-8
    out_min = 0
    out_max = 255
    shift_x = scale_param(20, img.shape, ref=400.0, min_val=0)
    shift_y = -scale_param(15, img.shape, ref=400.0, min_val=0)

    img_f = img.astype(np.float64)
    h, w = img_f.shape
    win_y = _create_1d_window(h, win_type, tukey_alpha)
    win_x = _create_1d_window(w, win_type, tukey_alpha)
    window_2d = np.outer(win_y, win_x)

    if dc_mode == "WINDOWED_MEAN":
        dc_val = np.sum(img_f * window_2d) / (np.sum(window_2d) + eps)
    else:
        dc_val = float(np.mean(img_f))

    zero_mean = img_f - dc_val
    windowed = zero_mean * window_2d

    if shift_x != 0 or shift_y != 0:
        M = np.float32([[1, 0, float(shift_x)], [0, 1, float(shift_y)]])
        windowed = cv2.warpAffine(windowed, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    std_val = float(np.std(windowed)) + eps
    normalized = windowed / (clip_sigma * std_val)
    normalized = np.clip(normalized, -1.0, 1.0)
    scaled = ((normalized + 1.0) * 0.5) * (out_max - out_min) + out_min
    return to_uint8(scaled)
