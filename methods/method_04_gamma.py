"""Метод 4. Гамма-преобразование: g = 255 * ((f-fmin)/(fmax-fmin))^gamma."""
import numpy as np
from ._utils import ensure_gray_uint8, to_uint8


def process(img):
    img = ensure_gray_uint8(img)
    gamma = 4.0  # из оригинала
    arr = img.astype(np.float64)
    mn, mx = float(arr.min()), float(arr.max())
    norm = (arr - mn) / (mx - mn + 1e-12)
    result = 255.0 * np.power(norm, gamma)
    return to_uint8(result)
