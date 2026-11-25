"""Метод 2. Частотный фильтр: Гауссов НЧ-срез в частотной области + растяжение."""
import numpy as np
import cv2
from ._utils import ensure_gray_uint8, to_uint8


def process(img):
    img = ensure_gray_uint8(img)
    sigma = 2.0  # из оригинала

    arr = img.astype(np.float64)
    h, w = arr.shape
    spectrum = np.fft.fftshift(np.fft.fft2(arr))
    u = np.arange(w) - w // 2
    v = np.arange(h) - h // 2
    U, V = np.meshgrid(u, v)
    rho = np.sqrt((U / float(w)) ** 2 + (V / float(h)) ** 2)
    H = np.exp(-2.0 * (np.pi * sigma * rho) ** 2)
    result = np.fft.ifft2(np.fft.ifftshift(spectrum * H)).real
    mn, mx = float(result.min()), float(result.max())
    result = (result - mn) / (mx - mn + 1e-12) * 255.0
    return to_uint8(result)
