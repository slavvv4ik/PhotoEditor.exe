# -*- coding: utf-8 -*-
"""Морфологическая скелетизация силуэта здания. Сегментация здания, заливка внутренних пустот и сведение его формы
 к остову (скелету) толщиной в несколько пикселей со смещением от центра."""

# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np


def process_skeletonization(
    img,
    blur_ksize,
    blur_sigma,
    roof_min,
    roof_max,
    close_ksize,
    open_ksize,
    skel_elem_size,
    skel_thickness,
    shift_x,
    shift_y,
    border_val,
):
    # Сглаживание и выделение яркостного диапазона крыши здания
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), blur_sigma)
    mask = cv2.inRange(blurred, roof_min, roof_max)

    # Морфологическое замыкание и размыкание для формирования сплошного силуэта
    k_close = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_ksize, close_ksize)
    )
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksize, open_ksize))
    solid = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, k_open)

    # Пошаговое построение морфологического скелета (остова) фигуры
    skeleton = np.zeros(solid.shape, np.uint8)
    element = cv2.getStructuringElement(
        cv2.MORPH_CROSS, (skel_elem_size, skel_elem_size)
    )
    temp_img = solid.copy()

    while True:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        temp_img = eroded.copy()
        if cv2.countNonZero(temp_img) == 0:
            break

    # Настройка итоговой толщины линий скелета
    if skel_thickness > 1:
        k_dilate = cv2.getStructuringElement(
            cv2.MORPH_RECT, (skel_thickness, skel_thickness)
        )
        skeleton = cv2.dilate(skeleton, k_dilate)

    # Сдвиг скелета от геометрического центра кадра
    h, w = img.shape
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_skeleton = cv2.warpAffine(
        skeleton,
        M,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )

    return shifted_skeleton


# ================================= НАСТРОЙКИ =================================
# Пути к файлам (поддерживается кириллица)
INPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_школа.png"
OUTPUT_IMAGE_PATH = "D:/ЭКСПЕРИМЕНТЫ/input_images/ЭИ_МС.png"

# Параметры преобразования
BLUR_KERNEL_SIZE = 15  # Размер ядра размытия перед сегментацией [диапазон: нечётные числа от 1 до 31]
BLUR_SIGMA = 0  # Сигма размытия Гаусса [диапазон: 0.0 ... 10.0; 0 - авторасчёт]
ROOF_BRIGHTNESS_MIN = 85  # Нижняя граница яркости крыши [диапазон: 0 ... 200, меньше ROOF_BRIGHTNESS_MAX]
ROOF_BRIGHTNESS_MAX = 145  # Верхняя граница яркости крыши [диапазон: 50 ... 255, больше ROOF_BRIGHTNESS_MIN]
MORPH_CLOSE_SIZE = 21  # Размер ядра для заливки пустот внутри контура [диапазон: нечётные числа от 3 до 51]
MORPH_OPEN_SIZE = 15  # Размер ядра для удаления внешнего шума [диапазон: нечётные числа от 3 до 51]
SKEL_CROSS_SIZE = 3  # Размер крестообразного элемента скелетизации [диапазон: нечётные числа 3, 5, 7]
SKELETON_THICKNESS = 3  # Итоговая толщина линий остова, пикс. [диапазон: целые числа от 1 до 10]
OFFSET_X = 35  # Смещение остова по горизонтали, пикс. [диапазон: -150 ... 150]
OFFSET_Y = 30  # Смещение остова по вертикали, пикс. [диапазон: -150 ... 150]
BACKGROUND_FILL_VALUE = 0  # Яркость фона при сдвиге [диапазон: 0 ... 255]

# Ввод исходного изображения (с поддержкой кириллицы в пути)
raw_bytes = np.fromfile(INPUT_IMAGE_PATH, dtype=np.uint8)
source_img = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)

# Вычисление обработанного изображения и вывод в файл
if source_img is not None:
    result_img = process_skeletonization(
        source_img,
        BLUR_KERNEL_SIZE,
        BLUR_SIGMA,
        ROOF_BRIGHTNESS_MIN,
        ROOF_BRIGHTNESS_MAX,
        MORPH_CLOSE_SIZE,
        MORPH_OPEN_SIZE,
        SKEL_CROSS_SIZE,
        SKELETON_THICKNESS,
        OFFSET_X,
        OFFSET_Y,
        BACKGROUND_FILL_VALUE,
    )
    ext = os.path.splitext(OUTPUT_IMAGE_PATH)[1]
    success, encoded_img = cv2.imencode(ext, result_img)
    if success:
        encoded_img.tofile(OUTPUT_IMAGE_PATH)
# =============================================================================