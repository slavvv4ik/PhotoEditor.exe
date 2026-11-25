# -*- coding: utf-8 -*-
"""Выделение локальных экстремумов (преобразование Top-Hat / «созвездие» опорных точек). Наиболее острые и
математически предсказуемые пики на полях дают компактные высококонтрастные элементы. Морфологическое
преобразование Top-Hat выделяет только мелкие контрастные детали конструкции (дымоходы, вентиляционные шахты,
 углы стыков крыши), превращая здание в чёткую матрицу точечных ориентиров на нулевом фоне."""

# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np


def process_tophat_keypoints(
    img,
    tophat_ksize,
    thresh_val,
    max_val,
    dilate_ksize,
    border_margin,
    shift_x,
    shift_y,
    border_val,
):
    # Структурный элемент для выделения компактных деталей (дымоходы, вентиляционные выступы, стыки)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (tophat_ksize, tophat_ksize)
    )

    # Выделение светлых и тёмных локальных экстремумов на крыше
    white_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    black_tophat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    combined = cv2.add(white_tophat, black_tophat)

    # Очистка периферии кадра от деревьев и кустарников
    h, w = combined.shape
    if border_margin > 0:
        combined[:border_margin, :] = 0
        combined[h - border_margin :, :] = 0
        combined[:, :border_margin] = 0
        combined[:, w - border_margin :] = 0

    # Отсечение слабых откликов по порогу
    _, keypoints = cv2.threshold(
        combined, thresh_val, max_val, cv2.THRESH_BINARY
    )

    # Укрупнение найденных опорных точек
    if dilate_ksize > 1:
        k_dil = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize)
        )
        keypoints = cv2.dilate(keypoints, k_dil)

    # Сдвиг матрицы опорных точек для устранения центральной симметрии
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_keypoints = cv2.warpAffine(
        keypoints,
        M,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )

    return shifted_keypoints


# ================================= НАСТРОЙКИ =================================
# Пути к файлам (поддерживается кириллица)
INPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_.png"
OUTPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_ЛЭ.png"

# Параметры преобразования
TOPHAT_KERNEL_SIZE = 19  # Размер структурного элемента фильтрации Top-Hat [диапазон: нечётные числа от 3 до 35]
EXTREMUM_THRESHOLD = 75  # Порог бинаризации локальных экстремумов [диапазон: 10 ... 200]
EXTREMUM_MAX_VALUE = 255  # Яркость выделенных ключевых точек [диапазон: 1 ... 255]
POINT_DILATION_SIZE = 3  # Диаметр укрупнения точек, пикс. [диапазон: целые числа от 1 до 15]
BORDER_CUTOFF_PIXELS = 45  # Ширина обнуляемой краевой зоны кадра, пикс. [диапазон: 0 ... 120]
SHIFT_X = 20  # Сдвиг по оси X, пикс. [диапазон: -150 ... 150]
SHIFT_Y = -25  # Сдвиг по оси Y, пикс. [диапазон: -150 ... 150]
BACKGROUND_FILL_VALUE = 0  # Яркость фона при сдвиге [диапазон: 0 ... 255]

# Ввод исходного изображения (с поддержкой кириллицы в пути)
raw_bytes = np.fromfile(INPUT_IMAGE_PATH, dtype=np.uint8)
source_img = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)

# Вычисление обработанного изображения и вывод в файл
if source_img is not None:
    result_img = process_tophat_keypoints(
        source_img,
        TOPHAT_KERNEL_SIZE,
        EXTREMUM_THRESHOLD,
        EXTREMUM_MAX_VALUE,
        POINT_DILATION_SIZE,
        BORDER_CUTOFF_PIXELS,
        SHIFT_X,
        SHIFT_Y,
        BACKGROUND_FILL_VALUE,
    )
    ext = os.path.splitext(OUTPUT_IMAGE_PATH)[1]
    success, encoded_img = cv2.imencode(ext, result_img)
    if success:
        encoded_img.tofile(OUTPUT_IMAGE_PATH)
# =============================================================================