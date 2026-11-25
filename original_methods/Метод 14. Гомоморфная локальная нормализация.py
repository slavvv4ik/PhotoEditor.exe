# -*- coding: utf-8 -*- """Гомоморфная локальная нормализация (логарифм, вычитание фона, деление на локальное СКО).
# Выравнивает тени и неравномерность освещения, поэтому структуры крыши дают сопоставимый вклад"""
import os
import cv2
import numpy as np


def process_homomorphic_norm(
    img,
    log_eps,
    bg_ksize,
    bg_sigma,
    std_ksize,
    std_sigma,
    std_reg,
    clip_limit,
    out_min,
    out_max,
    shift_x,
    shift_y,
):
    # 1. Логарифмирование исходного сигнала яркости
    img_f = img.astype(np.float64)
    log_img = np.log(img_f + log_eps)

    # 2. Оценка низкочастотного фона освещённости и его вычитание
    log_bg = cv2.GaussianBlur(log_img, (bg_ksize, bg_ksize), bg_sigma)
    high_pass = log_img - log_bg

    # 3. Вычисление локального СКО в скользящем окне
    local_mean = cv2.GaussianBlur(high_pass, (std_ksize, std_ksize), std_sigma)
    local_sq_mean = cv2.GaussianBlur(
        high_pass**2, (std_ksize, std_ksize), std_sigma
    )
    local_var = np.maximum(0.0, local_sq_mean - local_mean**2)
    local_std = np.sqrt(local_var)

    # 4. Деление на локальное СКО с регуляризацией
    normalized = (high_pass - local_mean) / (local_std + std_reg)

    # Опциональный сдвиг для нарушения центральной симметрии
    h, w = img.shape
    if shift_x != 0 or shift_y != 0:
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        normalized = cv2.warpAffine(
            normalized,
            M,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    # 5. Ограничение выбросов и приведение к выходному диапазону яркости
    clipped = np.clip(normalized, -clip_limit, clip_limit) / clip_limit
    scaled = ((clipped + 1.0) * 0.5) * (out_max - out_min) + out_min

    return np.round(scaled).astype(np.uint8)


# ================================= НАСТРОЙКИ =================================
# Пути к файлам (поддерживается кириллица)
INPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_школа.png"
OUTPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_ГЛН.png"

# Параметры обработки и диапазоны их изменения:
LOG_OFFSET_EPS = 1.0  # Смещение яркости перед логарифмом ln(I + eps) [диапазон: 0.01 ... 10.0]
BG_KERNEL_SIZE = 51  # Размер ядра Гаусса для оценки фона [диапазон: нечётные числа от 15 до 151]
BG_SIGMA = 20.0  # Сигма фильтра Гаусса для оценки фона [диапазон: 5.0 ... 80.0]
STD_KERNEL_SIZE = 21  # Размер локального окна для расчёта СКО [диапазон: нечётные числа от 5 до 75]
STD_SIGMA = 7.0  # Сигма окна для расчёта локального СКО [диапазон: 1.0 ... 30.0]
STD_REGULARIZATION = 0.05  # Добавка к локальному СКО для подавления шума фона [диапазон: 0.005 ... 0.5]
CLIP_SIGMA_LIMIT = 3.0  # Порог ограничения нормированного сигнала в единицах СКО [диапазон: 1.0 ... 6.0]
SHIFT_X = 20  # Сдвиг по горизонтали в пикселях [диапазон: -150 ... 150]
SHIFT_Y = -15  # Сдвиг по вертикали в пикселях [диапазон: -150 ... 150]
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
    result_img = process_homomorphic_norm(
        source_img,
        LOG_OFFSET_EPS,
        BG_KERNEL_SIZE,
        BG_SIGMA,
        STD_KERNEL_SIZE,
        STD_SIGMA,
        STD_REGULARIZATION,
        CLIP_SIGMA_LIMIT,
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