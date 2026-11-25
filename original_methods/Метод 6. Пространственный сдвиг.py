# -*- coding: utf-8 -*-
"""Пространственный сдвиг в квадрант с подавлением фона (нарушение центральной симметрии). Здание очищается от
среднетонового фона, масштабируется и смещается из центра кадра в один из квадрантов. Это разносит в фазовой
 плоскости пик автокорреляции и пик автосвёртки, исключая их взаимное уничтожение при вычитании."""


# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np


def process_spatial_shift(
    img,
    blur_ksize,
    blur_sigma,
    bg_low,
    bg_high,
    scale,
    shift_x,
    shift_y,
    contrast_alpha,
    contrast_beta,
):
    # Сглаживание мелкого текстурного шума
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), blur_sigma)

    # Обнуление среднетонового фона (оставляем только контрастные элементы и тени)
    mask = (blurred < bg_low) | (blurred > bg_high)
    cleaned = np.zeros_like(img)
    cleaned[mask] = img[mask]

    # Усиление контраста оставшихся элементов здания
    cleaned = cv2.convertScaleAbs(
        cleaned, alpha=contrast_alpha, beta=contrast_beta
    )

    # Масштабирование и сдвиг из центра в квадрант (нарушение центральной симметрии)
    h, w = img.shape
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cleaned, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((h, w), dtype=np.uint8)
    x_start = np.clip(shift_x, 0, w - new_w)
    y_start = np.clip(shift_y, 0, h - new_h)
    canvas[y_start : y_start + new_h, x_start : x_start + new_w] = resized

    return canvas


# ================================= НАСТРОЙКИ =================================
# Пути к файлам (поддерживается кириллица)
INPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_домик.png"
OUTPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_ПС.png"

# Параметры преобразования
BLUR_KERNEL_SIZE = 5  # Размер ядра размытия Гаусса [диапазон: нечётные числа от 1 до 31]
BLUR_SIGMA = 0  # Стандартное отклонение размытия Гаусса [диапазон: 0.0 ... 10.0; 0 - авторасчёт]
BG_THRESHOLD_LOW = 95  # Нижняя граница подавляемого серого фона [диапазон: 0 ... 254, меньше BG_THRESHOLD_HIGH]
BG_THRESHOLD_HIGH = 175  # Верхняя граница подавляемого серого фона [диапазон: 1 ... 255, больше BG_THRESHOLD_LOW]
SCALE_FACTOR = 0.55  # Коэффициент масштабирования здания [диапазон: 0.1 ... 1.0]
SHIFT_X_PIXELS = 180  # Смещение по горизонтали от левого края, пикс. [диапазон: 0 ... 400 (до края кадра)]
SHIFT_Y_PIXELS = 20  # Смещение по вертикали от верхнего края, пикс. [диапазон: 0 ... 400 (до края кадра)]
CONTRAST_GAIN = 1.4  # Коэффициент усиления контраста (alpha) [диапазон: 0.5 ... 3.0]
BRIGHTNESS_OFFSET = 0  # Смещение яркости при контрастировании (beta) [диапазон: -100 ... 100]

# Ввод исходного изображения (с поддержкой кириллицы в пути)
raw_bytes = np.fromfile(INPUT_IMAGE_PATH, dtype=np.uint8)
source_img = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)

# Вычисление обработанного изображения и вывод в файл
if source_img is not None:
    result_img = process_spatial_shift(
        source_img,
        BLUR_KERNEL_SIZE,
        BLUR_SIGMA,
        BG_THRESHOLD_LOW,
        BG_THRESHOLD_HIGH,
        SCALE_FACTOR,
        SHIFT_X_PIXELS,
        SHIFT_Y_PIXELS,
        CONTRAST_GAIN,
        BRIGHTNESS_OFFSET,
    )
    ext = os.path.splitext(OUTPUT_IMAGE_PATH)[1]
    success, encoded_img = cv2.imencode(ext, result_img)
    if success:
        encoded_img.tofile(OUTPUT_IMAGE_PATH)
# =============================================================================