#"""Адаптивная бинаризация в симметричный сигнал ±1 с морфологической очисткой.
# Все структуры получают одинаковую амплитуду, шумы убраны"""
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np


def process_bipolar_adaptive_bin(
    img,
    blur_ksize,
    blur_sigma,
    block_size,
    c_const,
    invert,
    open_ksize,
    close_ksize,
    shift_x,
    shift_y,
    level_minus_one,
    level_plus_one,
):
    # Предварительное сглаживание
    if blur_ksize > 1:
        smoothed = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), blur_sigma)
    else:
        smoothed = img.copy()

    # Адаптивная пороговая бинаризация
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    binary = cv2.adaptiveThreshold(
        smoothed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresh_type,
        block_size,
        c_const,
    )

    # Морфологическая очистка: открытие (удаление шума) и закрытие (заполнение разрывов)
    cleaned = binary
    if open_ksize > 1:
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_open)
    if close_ksize > 1:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    # Перевод в симметричный биполярный сигнал {-1.0, +1.0}
    bipolar_signal = np.where(cleaned > 0, 1.0, -1.0)

    # Сдвиг для устранения центральной симметрии (фон заполняется уровнем -1.0)
    h, w = img.shape
    if shift_x != 0 or shift_y != 0:
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        bipolar_signal = cv2.warpAffine(
            bipolar_signal,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1.0,
        )

    # Кодирование уровней -1 и +1 в диапазон яркости изображения для сохранения
    output_img = np.where(bipolar_signal > 0.0, level_plus_one, level_minus_one)
    return output_img.astype(np.uint8)


# ================================= НАСТРОЙКИ =================================
# Пути к файлам (поддерживается кириллица)
INPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_школа.png"
OUTPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_АБ.png"

# Параметры обработки и диапазоны их изменения:
PRE_BLUR_KSIZE = 5  # Размер ядра предварительного размытия [диапазон: нечётные числа от 1 до 15; 1 - выкл.]
PRE_BLUR_SIGMA = 0.0  # Сигма предварительного размытия [диапазон: 0.0 ... 5.0]
ADAPTIVE_BLOCK_SIZE = 25  # Размер локальной окрестности адаптивного порога [диапазон: нечётные числа от 3 до 99]
ADAPTIVE_C_CONSTANT = (
    4.0  # Вычитаемая константа адаптивного порога [диапазон: -25.0 ... 25.0]
)
INVERT_POLARITY = True  # Инверсия знаков (+1 для тёмных структур и теней здания) [диапазон: True или False]
MORPH_OPEN_KSIZE = 3  # Размер ядра морфологического открытия для очистки от мелких точек [диапазон: 1 ... 15]
MORPH_CLOSE_KSIZE = 3  # Размер ядра морфологического закрытия для слияния контуров [диапазон: 1 ... 15]
SHIFT_X = 25  # Сдвиг по горизонтали в пикселях [диапазон: -150 ... 150]
SHIFT_Y = -20  # Сдвиг по вертикали в пикселях [диапазон: -150 ... 150]
LEVEL_MINUS_ONE = 0  # Яркость пикселя в файле, кодирующая уровень -1 [диапазон: 0 ... 127]
LEVEL_PLUS_ONE = 255  # Яркость пикселя в файле, кодирующая уровень +1 [диапазон: 128 ... 255]

# Ввод исходного изображения
raw_bytes = np.fromfile(INPUT_IMAGE_PATH, dtype=np.uint8)
source_img = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)

# Обработка и вывод результата
if source_img is not None:
    result_img = process_bipolar_adaptive_bin(
        source_img,
        PRE_BLUR_KSIZE,
        PRE_BLUR_SIGMA,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C_CONSTANT,
        INVERT_POLARITY,
        MORPH_OPEN_KSIZE,
        MORPH_CLOSE_KSIZE,
        SHIFT_X,
        SHIFT_Y,
        LEVEL_MINUS_ONE,
        LEVEL_PLUS_ONE,
    )
    ext = os.path.splitext(OUTPUT_IMAGE_PATH)[1]
    success, encoded_img = cv2.imencode(ext, result_img)
    if success:
        encoded_img.tofile(OUTPUT_IMAGE_PATH)
# =============================================================================