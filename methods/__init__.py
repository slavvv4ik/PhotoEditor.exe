from .method_01_otsu import process as m01
from .method_02_frequency import process as m02
from .method_03_morphology import process as m03
from .method_04_gamma import process as m04
from .method_05_matched_filter import process as m05
from .method_06_spatial_shift import process as m06
from .method_07_structural_skeleton import process as m07
from .method_08_morph_skeleton import process as m08
from .method_09_asymmetric_highlight import process as m09
from .method_10_local_extrema import process as m10
from .method_11_binarization import process as m11
from .method_12_dc_window import process as m12
from .method_13_log_gradient import process as m13
from .method_14_homomorphic import process as m14
from .method_15_adaptive_bipolar import process as m15
from .method_16_segment_scale_shift import process as m16

METHODS = [
    ("01. Порог Отсу", m01),
    ("02. Частотный фильтр", m02),
    ("03. Морфология", m03),
    ("04. Гамма-преобразование", m04),
    ("05. Согласованный фильтр", m05),
    ("06. Пространственный сдвиг", m06),
    ("07. Структурный каркас", m07),
    ("08. Морфологическая скелетизация", m08),
    ("09. Асимметричное выделение", m09),
    ("10. Локальные экстремумы", m10),
    ("11. Бинаризация", m11),
    ("12. Вычитание пост. сост. + окно", m12),
    ("13. LoG/Градиент", m13),
    ("14. Гомоморфная локальная нормализация", m14),
    ("15. Адаптивная бинаризация (биполярная)", m15),
    ("16. Сегментация, масштабирование", m16),
]
