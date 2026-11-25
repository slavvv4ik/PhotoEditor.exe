# -*- coding: utf-8 -*-
"""Выделение структурного каркаса крыши (контурная фильтрация). Сглаживание текстуры травы и деревьев с последующим
 выделением границ (рёбер крыши и стен) детектором Канни и морфологическим утолщением. Превращает массивное здание
 в линейчатый геометрический паттерн, который идеально читается по изолиниям МРХ."""


import os
import cv2
import numpy as np


def process_structural_edges(
    img,
    bilateral_d,
    sigma_color,
    sigma_space,
    canny_low,
    canny_high,
    dilate_size,
    shift_x,
    shift_y,
    border_val,
):
    # Сглаживание текстуры травы и бетона с сохранением чётких граней здания
    filtered = cv2.bilateralFilter(img, bilateral_d, sigma_color, sigma_space)

    # Детектирование границ (каркас крыши и контуры стен)
    edges = cv2.Canny(filtered, canny_low, canny_high)

    # Морфологическое утолщение контурных линий
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (dilate_size, dilate_size)
    )
    thick_edges = cv2.dilate(edges, kernel)

    # Асимметричный сдвиг контурного каркаса относительно центра кадра
    h, w = img.shape
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_edges = cv2.warpAffine(
        thick_edges,
        M,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )

    return shifted_edges


# ================================= НАСТРОЙКИ =================================
# Пути к файлам (поддерживается кириллица)
INPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_домик.png"
OUTPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_СК.png"

# Параметры преобразования
BILATERAL_DIAMETER = 10  # Диаметр окрестности билатерального фильтра [диапазон: целые числа от 3 до 25]
SIGMA_COLOR = 75  #  Параметр фильтрации в пространстве яркости [диапазон: 10.0 ... 200.0]
SIGMA_SPACE = 75  # Параметр фильтрации в пространстве координат [диапазон: 10.0 ... 200.0]
CANNY_LOWER_THRESHOLD = 60  # Нижний порог детектора границ Канни [диапазон: 10 ... 150, меньше CANNY_UPPER_THRESHOLD]
CANNY_UPPER_THRESHOLD = 160  # Верхний порог детектора границ Канни [диапазон: 50 ... 300, больше CANNY_LOWER_THRESHOLD]
LINE_THICKNESS_KERNEL = 2  # Размер ядра утолщения контуров, пикс. [диапазон: целые числа от 1 до 10]
ASYMMETRY_SHIFT_X = 25  # Сдвиг каркаса по оси X, пикс. [диапазон: -150 ... 150]
ASYMMETRY_SHIFT_Y = -20  # Сдвиг каркаса по оси Y, пикс. [диапазон: -150 ... 150]
BACKGROUND_FILL_VALUE = 0  # Яркость заполнения фона при сдвиге [диапазон: 0 ... 255]

# Ввод исходного изображения (с поддержкой кириллицы в пути)
raw_bytes = np.fromfile(INPUT_IMAGE_PATH, dtype=np.uint8)
source_img = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)

# Вычисление обработанного изображения и вывод в файл
if source_img is not None:
    result_img = process_structural_edges(
        source_img,
        BILATERAL_DIAMETER,
        SIGMA_COLOR,
        SIGMA_SPACE,
        CANNY_LOWER_THRESHOLD,
        CANNY_UPPER_THRESHOLD,
        LINE_THICKNESS_KERNEL,
        ASYMMETRY_SHIFT_X,
        ASYMMETRY_SHIFT_Y,
        BACKGROUND_FILL_VALUE,
    )
    ext = os.path.splitext(OUTPUT_IMAGE_PATH)[1]
    success, encoded_img = cv2.imencode(ext, result_img)
    if success:
        encoded_img.tofile(OUTPUT_IMAGE_PATH)
# =============================================================================