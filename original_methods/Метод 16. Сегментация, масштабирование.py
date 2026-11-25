# -*- coding: utf-8 -*-"""Сегментация здания, обнуление фона, масштабирование и смещение от центра кадра.
# Фон равен нулю. Отклик не сливается с центральным минимумом и не гасится симметрией"""
import os
import cv2
import numpy as np


def process_segment_scale_shift(
    img,
    blur_ksize,
    blur_sigma,
    seg_min,
    seg_max,
    border_margin,
    close_ksize,
    open_ksize,
    keep_largest,
    contrast_alpha,
    scale_factor,
    offset_x,
    offset_y,
):
    h, w = img.shape

    # Сглаживание для устойчивой сегментации силуэта здания
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), blur_sigma)

    # Пороговая сегментация здания и прилегающих теней (отсечение светлого бетона и газонов)
    mask = cv2.inRange(blurred, seg_min, seg_max)

    # Принудительное обнуление краевой рамки (отсечение дорог и деревьев на границах кадра)
    if border_margin > 0:
        mask[:border_margin, :] = 0
        mask[h - border_margin :, :] = 0
        mask[:, :border_margin] = 0
        mask[:, w - border_margin :] = 0

    # Морфологическая очистка маски здания
    k_close = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_ksize, close_ksize)
    )
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksize, open_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # Выделение только самого крупного связного объекта (здания)
    if keep_largest:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # Наложение маски на исходное изображение (полное обнуление фона)
    segmented = np.zeros_like(img)
    segmented[mask > 0] = img[mask > 0]
    segmented = cv2.convertScaleAbs(segmented, alpha=contrast_alpha, beta=0)

    # Масштабирование выделенного здания
    new_w = max(1, int(w * scale_factor))
    new_h = max(1, int(h * scale_factor))
    resized = cv2.resize(
        segmented, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    # Размещение со смещением относительно геометрического центра кадра
    canvas = np.zeros((h, w), dtype=np.uint8)
    center_x = (w - new_w) // 2 + offset_x
    center_y = (h - new_h) // 2 + offset_y

    x_start = int(np.clip(center_x, 0, w - new_w))
    y_start = int(np.clip(center_y, 0, h - new_h))
    canvas[y_start : y_start + new_h, x_start : x_start + new_w] = resized

    return canvas


# ================================= НАСТРОЙКИ =================================
# Пути к файлам (поддерживается кириллица)
INPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_школа.png"
OUTPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_СМ.png"

# Параметры обработки и диапазоны их изменения:
BLUR_KSIZE = 7  # Размер ядра размытия перед сегментацией [диапазон: нечётные числа от 3 до 25]
BLUR_SIGMA = 0.0  # Сигма размытия Гаусса [диапазон: 0.0 ... 10.0]
BUILDING_INTENSITY_MIN = 10  # Нижняя граница яркости здания и его теней [диапазон: 0 ... 120]
BUILDING_INTENSITY_MAX = (
    148  # Верхняя граница яркости скатов крыши [диапазон: 110 ... 220]
)
BORDER_CROP_MARGIN = 35  # Ширина обнуляемой рамки у краёв кадра, пикс. [диапазон: 0 ... 100]
MORPH_CLOSE_KSIZE = 25  # Размер ядра замыкания для получения монолитной маски здания [диапазон: нечётные 3 ... 55]
MORPH_OPEN_KSIZE = 15  # Размер ядра размыкания для удаления деревьев вокруг здания [диапазон: нечётные 3 ... 35]
KEEP_LARGEST_OBJECT = True  # Оставить только крупнейший связный объект (само здание) [диапазон: True или False]
CONTRAST_GAIN = (
    1.3  # Коэффициент усиления яркости текстуры крыши [диапазон: 0.5 ... 3.0]
)
SCALE_FACTOR = 0.55  # Коэффициент масштабирования здания [диапазон: 0.25 ... 1.0]
CENTER_OFFSET_X = 75  # Смещение от центра кадра по горизонтали, пикс. [диапазон: -180 ... 180]
CENTER_OFFSET_Y = -65  # Смещение от центра кадра по вертикали, пикс. [диапазон: -180 ... 180]

# Ввод исходного изображения
raw_bytes = np.fromfile(INPUT_IMAGE_PATH, dtype=np.uint8)
source_img = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)

# Обработка и вывод результата
if source_img is not None:
    result_img = process_segment_scale_shift(
        source_img,
        BLUR_KSIZE,
        BLUR_SIGMA,
        BUILDING_INTENSITY_MIN,
        BUILDING_INTENSITY_MAX,
        BORDER_CROP_MARGIN,
        MORPH_CLOSE_KSIZE,
        MORPH_OPEN_KSIZE,
        KEEP_LARGEST_OBJECT,
        CONTRAST_GAIN,
        SCALE_FACTOR,
        CENTER_OFFSET_X,
        CENTER_OFFSET_Y,
    )
    ext = os.path.splitext(OUTPUT_IMAGE_PATH)[1]
    success, encoded_img = cv2.imencode(ext, result_img)
    if success:
        encoded_img.tofile(OUTPUT_IMAGE_PATH)
# =============================================================================