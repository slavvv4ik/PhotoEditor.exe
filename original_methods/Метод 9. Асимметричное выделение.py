# -*- coding: utf-8 -*-
"""Асимметричное выделение теней и бликов (естественное нарушение симметрии без сдвига). Само здание симметрично,
но падающие от него тени — асимметричны (солнце светит под углом). Если обнулить все средние тона (бетон, траву,
 скаты крыши), а глубокие тени инвертировать в яркие области вместе с яркими коньками крыши, симметрия разрушится
 естественным путём без изменения координат здания."""

import os
import cv2
import numpy as np


def process_asymmetric_shadows(
    img,
    median_ksize,
    shadow_thresh,
    highlight_thresh,
    shadow_intensity,
    highlight_intensity,
    border_crop,
):
    # Подавление мелкого шума медианным фильтром
    smoothed = cv2.medianBlur(img, median_ksize)

    # Формирование пустого поля с нулевым фоном
    output = np.zeros_like(img)

    # Выделение падающих теней (естественный несимметричный контур здания)
    output[smoothed <= shadow_thresh] = shadow_intensity

    # Выделение освещённых коньков и карнизов крыши
    output[smoothed >= highlight_thresh] = highlight_intensity

    # Обнуление краевой зоны кадра (удаление деревьев и дорог по периметру)
    h, w = output.shape
    if border_crop > 0:
        output[:border_crop, :] = 0
        output[h - border_crop :, :] = 0
        output[:, :border_crop] = 0
        output[:, w - border_crop :] = 0

    return output


# ================================= НАСТРОЙКИ =================================
# Пути к файлам (поддерживается кириллица)
INPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_.png"
OUTPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_АВ.png"

# Параметры преобразования
MEDIAN_BLUR_SIZE = 15  # Размер апертуры медианного фильтра [диапазон: нечётные числа от 3 до 25]
SHADOW_MAX_BRIGHTNESS = 55  # Верхний порог яркости для выделения глубоких теней [диапазон: 5 ... 110]
HIGHLIGHT_MIN_BRIGHTNESS = 215  # Нижний порог яркости для выделения светлых рёбер крыши [диапазон: 150 ... 250]
OUTPUT_SHADOW_VALUE = 255  # Присваиваемая яркость областям теней [диапазон: 1 ... 255]
OUTPUT_HIGHLIGHT_VALUE = 140  # Присваиваемая яркость светлым карнизам [диапазон: 1 ... 255]
PERIMETER_MASK_WIDTH = 35  # Ширина обнуляемой рамки у краёв кадра, пикс. [диапазон: 0 ... 120]

# Ввод исходного изображения (с поддержкой кириллицы в пути)
raw_bytes = np.fromfile(INPUT_IMAGE_PATH, dtype=np.uint8)
source_img = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)

# Вычисление обработанного изображения и вывод в файл
if source_img is not None:
    result_img = process_asymmetric_shadows(
        source_img,
        MEDIAN_BLUR_SIZE,
        SHADOW_MAX_BRIGHTNESS,
        HIGHLIGHT_MIN_BRIGHTNESS,
        OUTPUT_SHADOW_VALUE,
        OUTPUT_HIGHLIGHT_VALUE,
        PERIMETER_MASK_WIDTH,
    )
    ext = os.path.splitext(OUTPUT_IMAGE_PATH)[1]
    success, encoded_img = cv2.imencode(ext, result_img)
    if success:
        encoded_img.tofile(OUTPUT_IMAGE_PATH)
# =============================================================================
