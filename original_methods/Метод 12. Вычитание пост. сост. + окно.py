# -*- coding: utf-8 -*-
# """Вычитание постоянной составляющей, окно Тьюки или Ханна, нормировка.
# Убирает «плато» и краевые эффекты конечного кадра""""
import os
import cv2
import numpy as np


def create_1d_window(length, win_type, tukey_alpha):
    x = np.linspace(0.0, 1.0, length)
    if win_type == "HANN" or tukey_alpha >= 1.0:
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * x))
    elif tukey_alpha <= 0.0:
        return np.ones(length, dtype=np.float64)
    else:
        win = np.ones(length, dtype=np.float64)
        half_alpha = tukey_alpha / 2.0

        mask1 = x < half_alpha
        win[mask1] = 0.5 * (
            1.0 + np.cos(np.pi * (2.0 * x[mask1] / tukey_alpha - 1.0))
        )

        mask2 = x >= (1.0 - half_alpha)
        win[mask2] = 0.5 * (
            1.0
            + np.cos(
                np.pi * (2.0 * x[mask2] / tukey_alpha - 2.0 / tukey_alpha + 1.0)
            )
        )
        return win


def process_dc_window_norm(
    img,
    win_type,
    tukey_alpha,
    dc_mode,
    clip_sigma,
    eps,
    out_min,
    out_max,
    shift_x,
    shift_y,
):
    img_f = img.astype(np.float64)
    h, w = img_f.shape

    # Построение двумерного весового окна (Тьюки или Ханна)
    win_y = create_1d_window(h, win_type, tukey_alpha)
    win_x = create_1d_window(w, win_type, tukey_alpha)
    window_2d = np.outer(win_y, win_x)

    # Вычитание постоянной составляющей (среднего значения)
    if dc_mode == "WINDOWED_MEAN":
        dc_val = np.sum(img_f * window_2d) / (np.sum(window_2d) + eps)
    else:
        dc_val = np.mean(img_f)

    zero_mean = img_f - dc_val

    # Применение аподизирующего окна для подавления краевых скачков
    windowed = zero_mean * window_2d

    # Опциональный сдвиг для нарушения центральной симметрии
    if shift_x != 0 or shift_y != 0:
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        windowed = cv2.warpAffine(
            windowed,
            M,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    # Нормировка по среднеквадратическому отклонению (СКО) и перевод в выходной диапазон
    std_val = np.std(windowed) + eps
    normalized = windowed / (clip_sigma * std_val)
    normalized = np.clip(normalized, -1.0, 1.0)

    scaled = ((normalized + 1.0) * 0.5) * (out_max - out_min) + out_min
    return np.round(scaled).astype(np.uint8)


# ================================= НАСТРОЙКИ =================================
# Пути к файлам (поддерживается кириллица)
INPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_школа.png"
OUTPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_окно_нормировка.png"

# Параметры обработки и диапазоны их изменения:
WINDOW_TYPE = "TUKEY"  # Тип окна: 'TUKEY' (Тьюки) или 'HANN' (Ханна)
TUKEY_ALPHA = 0.35  # Коэффициент скругления окна Тьюки [диапазон: 0.0 ... 1.0; 0.0 - прямоугольное, 1.0 - Ханна]
DC_REMOVAL_MODE = "WINDOWED_MEAN"  # Способ расчёта постоянной составляющей: 'MEAN' (по кадру) или 'WINDOWED_MEAN' (взвешенное окном)
NORM_CLIP_SIGMA = 3.0  # Порог ограничения выбросов в долях СКО при нормировке [диапазон: 1.0 ... 8.0]
REGULARIZATION_EPS = (
    1e-8  # Малая константа защиты от деления на ноль [диапазон: 1e-12 ... 1e-4]
)
SHIFT_X = 20  # Сдвиг поля по горизонтали в пикселях для асимметрии [диапазон: -150 ... 150]
SHIFT_Y = -15  # Сдвиг поля по вертикали в пикселях для асимметрии [диапазон: -150 ... 150]
OUTPUT_MIN_LEVEL = (
    0  # Минимальная яркость выходного изображения [диапазон: 0 ... 127]
)
OUTPUT_MAX_LEVEL = (
    255  # Максимальная яркость выходного изображения [диапазон: 128 ... 255]
)

# Ввод исходного изображения
raw_bytes = np.fromfile(INPUT_IMAGE_PATH, dtype=np.uint8)
source_img = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)

# Обработка и вывод результата
if source_img is not None:
    result_img = process_dc_window_norm(
        source_img,
        WINDOW_TYPE,
        TUKEY_ALPHA,
        DC_REMOVAL_MODE,
        NORM_CLIP_SIGMA,
        REGULARIZATION_EPS,
        OUTPUT_MIN_LEVEL,
        OUTPUT_MAX_LEVEL,
        SHIFT_X,
        SHIFT_Y,
    )
    ext = os.path.splitext(OUTPUT_IMAGE_PATH)[1]
    success, encoded_img = cv2.imencode(ext, result_img)
    if success:
        encoded_img.tofile(OUTPUT_IMAGE_PATH)
# =============================================================================
