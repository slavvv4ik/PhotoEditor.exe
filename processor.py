import os
import cv2
import numpy as np
from methods import METHODS
from methods._utils import contrast_score, ensure_gray_uint8


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_image_gray(path):
    """Load image as grayscale uint8. Supports Unicode paths on Windows."""
    # cv2.imdecode + fromfile handles Cyrillic paths reliably on Windows
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    except Exception:
        pass
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {path}")
    return img


def save_image_gray(path, img_gray):
    """Save grayscale uint8 image. Supports Unicode paths on Windows."""
    img_gray = ensure_gray_uint8(img_gray)
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        ext = ".png"
        path = path + ext
    ok, buf = cv2.imencode(ext, img_gray)
    if not ok:
        raise ValueError(f"Не удалось закодировать изображение: {path}")
    buf.tofile(path)
    return path


def process_image(img_gray, include_scores=False):
    """
    Run all methods, pick result with highest contrast (std dev of luminance).
    Returns (best_img, best_method_name, best_score, scores).
    scores is list of dicts: {name, score, ok, error}
    """
    img_gray = ensure_gray_uint8(img_gray)
    scores = []
    best_img = None
    best_name = None
    best_score = -1.0

    for name, fn in METHODS:
        try:
            out = fn(img_gray.copy())
            out = ensure_gray_uint8(out)
            # Ensure same shape as input for consistent comparison/display
            if out.shape[:2] != img_gray.shape[:2]:
                out = cv2.resize(out, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_AREA)
            score = contrast_score(out)
            scores.append({"name": name, "score": score, "ok": True, "error": ""})
            if score > best_score:
                best_score = score
                best_img = out
                best_name = name
        except Exception as e:
            scores.append({"name": name, "score": None, "ok": False, "error": str(e)})

    if best_img is None:
        # Fallback: return original
        best_img = img_gray.copy()
        best_name = "Исходное (без обработки)"
        best_score = contrast_score(best_img)

    return best_img, best_name, best_score, scores


def is_supported_image(path):
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTS
