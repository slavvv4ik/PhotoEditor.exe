# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np


def convert_to_binary(
    img,
    blur_ksize,
    blur_sigma,
    method,
    manual_thresh,
    max_val,
    block_size,
    c_const,
    invert,
):
    # Предварительное сглаживание перед пороговой обработкой
    if blur_ksize > 1:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), blur_sigma)

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    if method == "OTSU":
        _, binary = cv2.threshold(
            img, manual_thresh, max_val, thresh_type + cv2.THRESH_OTSU
        )
    elif method == "ADAPTIVE":
        binary = cv2.adaptiveThreshold(
            img,
            max_val,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type,
            block_size,
            c_const,
        )
    else:  # Режим 'MANUAL'
        _, binary = cv2.threshold(img, manual_thresh, max_val, thresh_type)

    return binary


# ================================= НАСТРОЙКИ =================================
# Пути к файлам (поддерживается кириллица)
INPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_школа.png"
OUTPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_Б.png"

# Параметры бинаризации
BLUR_KERNEL_SIZE = 13  # Размер ядра предварительного размытия [диапазон: нечётные числа от 1 до 31; 1 - без размытия]
BLUR_SIGMA = 0  # Сигма размытия Гаусса [диапазон: 0.0 ... 10.0; 0 - авторасчёт]
BINARIZATION_METHOD = "OTSU"  # Метод бинаризации [допустимые значения: 'MANUAL', 'OTSU', 'ADAPTIVE']
MANUAL_THRESHOLD_VALUE = 127  # Порог яркости для режима 'MANUAL' [диапазон: 0 ... 255]
MAX_BINARY_VALUE = 255  # Значение яркости для белых пикселей [диапазон: 1 ... 255]
ADAPTIVE_BLOCK_SIZE = 15  # Размер окрестности для режима 'ADAPTIVE' [диапазон: нечётные числа от 3 до 99]
ADAPTIVE_C_CONSTANT = 4  # Вычитаемая константа для режима 'ADAPTIVE' [диапазон: -30 ... 30]
INVERT_RESULT = False  # Инверсия цветов [допустимые значения: True - инвертировать, False - обычная]

# Ввод исходного изображения (с поддержкой кириллицы в пути)
raw_bytes = np.fromfile(INPUT_IMAGE_PATH, dtype=np.uint8)
source_img = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)

# Вычисление бинарного изображения и вывод в файл
if source_img is not None:
    binary_img = convert_to_binary(
        source_img,
        BLUR_KERNEL_SIZE,
        BLUR_SIGMA,
        BINARIZATION_METHOD,
        MANUAL_THRESHOLD_VALUE,
        MAX_BINARY_VALUE,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C_CONSTANT,
        INVERT_RESULT,
    )
    ext = os.path.splitext(OUTPUT_IMAGE_PATH)[1]
    success, encoded_img = cv2.imencode(ext, binary_img)
    if success:
        encoded_img.tofile(OUTPUT_IMAGE_PATH)
# =============================================================================