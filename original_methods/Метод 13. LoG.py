# -*- coding: utf-8 -*-
#"""Контурный препарат: LoG или градиент с мягким порогом.
# Оставляет только границы крыш и элементов. Результат знакопеременный, почти без постоянной составляющей.
# Подавляет медленные изменения яркости"""

import os
import cv2
import numpy as np


def process_contour_prep(
    img,
    op_type,
    blur_ksize,
    blur_sigma,
    deriv_ksize,
    soft_thresh_ratio,
    soft_steepness,
    eps,
    shift_x,
    shift_y,
    out_max,
):
    img_f = img.astype(np.float64)

    # Предварительное гауссово сглаживание
    smoothed = cv2.GaussianBlur(img_f, (blur_ksize, blur_ksize), blur_sigma)

    # Вычисление отклика оператора (Градиент или Лапласиан Гауссиана - LoG)
    if op_type == "LOG":
        lap = cv2.Laplacian(smoothed, cv2.CV_64F, ksize=deriv_ksize)
        response = np.abs(lap)
    else:  # 'GRADIENT'
        gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=deriv_ksize)
        gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=deriv_ksize)
        response = np.hypot(gx, gy)

    # Нормировка отклика к интервалу [0, 1]
    max_resp = np.max(response) + eps
    norm_resp = response / max_resp

    # Мягкая пороговая обработка (Soft Thresholding с плавным насыщением)
    above_thresh = np.maximum(0.0, norm_resp - soft_thresh_ratio) / (
        1.0 - soft_thresh_ratio + eps
    )
    soft_contour = np.tanh(soft_steepness * above_thresh) / np.tanh(
        soft_steepness + eps
    )

    # Сдвиг контурного препарата для разрушения центральной симметрии
    h, w = img.shape
    if shift_x != 0 or shift_y != 0:
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        soft_contour = cv2.warpAffine(
            soft_contour,
            M,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    return np.round(soft_contour * out_max).astype(np.uint8)


# ================================= НАСТРОЙКИ =================================
# Пути к файлам (поддерживается кириллица)
INPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_школа.png"
OUTPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_LoG.png"

# Параметры обработки и диапазоны их изменения:
OPERATOR_TYPE = "GRADIENT"  # Тип оператора: 'GRADIENT' (модуль градиента Собеля) или 'LOG' (лапласиан гауссиана)
GAUSSIAN_KSIZE = 5  # Размер ядра сглаживания Гаусса [диапазон: нечётные числа от 3 до 31]
GAUSSIAN_SIGMA = (
    1.4  # Стандартное отклонение фильтра Гаусса [диапазон: 0.5 ... 10.0]
)
DERIVATIVE_KSIZE = 3  # Размер апертуры дифференциального оператора [диапазон: нечётные числа 1, 3, 5, 7]
SOFT_THRESHOLD_RATIO = 0.15  # Уровень мягкого порога в долях от максимума [диапазон: 0.0 ... 0.85]
SOFT_STEEPNESS = 2.5  # Крутизна функции мягкого насыщения (гиперболический тангенс) [диапазон: 0.5 ... 10.0]
REGULARIZATION_EPS = 1e-8  # Константа защиты от деления на ноль [диапазон: 1e-12 ... 1e-4]
SHIFT_X = 25  # Сдвиг препарата по горизонтали в пикселях [диапазон: -150 ... 150]
SHIFT_Y = -20  # Сдвиг препарата по вертикали в пикселях [диапазон: -150 ... 150]
OUTPUT_MAX_LEVEL = (
    255  # Максимальная яркость выходного изображения [диапазон: 1 ... 255]
)

# Ввод исходного изображения
raw_bytes = np.fromfile(INPUT_IMAGE_PATH, dtype=np.uint8)
source_img = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)

# Обработка и вывод результата
if source_img is not None:
    result_img = process_contour_prep(
        source_img,
        OPERATOR_TYPE,
        GAUSSIAN_KSIZE,
        GAUSSIAN_SIGMA,
        DERIVATIVE_KSIZE,
        SOFT_THRESHOLD_RATIO,
        SOFT_STEEPNESS,
        REGULARIZATION_EPS,
        SHIFT_X,
        SHIFT_Y,
        OUTPUT_MAX_LEVEL,
    )
    ext = os.path.splitext(OUTPUT_IMAGE_PATH)[1]
    success, encoded_img = cv2.imencode(ext, result_img)
    if success:
        encoded_img.tofile(OUTPUT_IMAGE_PATH)
# =============================================================================