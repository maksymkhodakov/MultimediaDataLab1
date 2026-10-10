"""
Спільні налаштування лабораторної роботи №1 «Класифікація звуків».

Тут зібрано всі шляхи до файлів та параметри перетворення звуку на спектрограму,
щоб усі скрипти (підготовка даних, навчання, прогноз, побудова звіту) працювали
з однаковими значеннями. Якщо змінити параметр тут — він зміниться всюди.
"""

from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Шляхи
# ---------------------------------------------------------------------------

# Корінь проєкту — тека на рівень вище за src/
ROOT_DIR = Path(__file__).resolve().parent.parent

# Тека з даними: сирі завантаження (тимчасово) та готові ознаки (спектрограми)
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # сюди тимчасово завантажуються Parquet / tar.gz
FEATURES_DIR = DATA_DIR / "features"  # сюди зберігаються готові спектрограми (.npz)

# Тека з власними записами міських звуків (пункт 4 завдання)
MY_SOUNDS_DIR = ROOT_DIR / "my_sounds"

# Тека з результатами: ваги моделей, метрики (JSON) та рисунки для звіту
RUNS_DIR = ROOT_DIR / "src" / "runs"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Файли з готовими ознаками
US8K_FEATURES = FEATURES_DIR / "urbansound8k_logmel.npz"
GTZAN_FEATURES = FEATURES_DIR / "gtzan_logmel.npz"

# ---------------------------------------------------------------------------
# Параметри перетворення звуку на log-mel спектрограму
# ---------------------------------------------------------------------------

# Частота дискретизації, до якої приводимо всі записи (Гц).
# 22 050 Гц — стандарт librosa; цього досить для міських звуків і музики
# (за теоремою Котельникова передаються частоти до 11 025 Гц).
SAMPLE_RATE = 22050

# Розмір вікна швидкого перетворення Фур'є (кількість відліків в одному кадрі).
# 2048 відліків ≈ 93 мс — компроміс між роздільністю за частотою та за часом.
N_FFT = 2048

# Крок між сусідніми кадрами (відліків). 512 відліків ≈ 23 мс, тобто вікна
# перекриваються на 75 %.
HOP_LENGTH = 512

# --- UrbanSound8K ---
US8K_CLIP_SECONDS = 4.0  # максимальна тривалість кліпу в датасеті — 4 с
US8K_N_MELS = 64         # кількість mel-смуг (висота «картинки»)
US8K_CLIP_SAMPLES = int(SAMPLE_RATE * US8K_CLIP_SECONDS)  # 88 200 відліків
# Кількість часових кадрів: 1 + 88200 // 512 = 173 (при center=True)
US8K_FRAMES = 1 + US8K_CLIP_SAMPLES // HOP_LENGTH

# Назви 10 класів UrbanSound8K у порядку їхніх номерів classID
US8K_CLASSES = [
    "air_conditioner",   # 0 — кондиціонер
    "car_horn",          # 1 — автомобільний сигнал
    "children_playing",  # 2 — дитячі ігри
    "dog_bark",          # 3 — гавкіт собаки
    "drilling",          # 4 — дриль
    "engine_idling",     # 5 — двигун на холостому ходу
    "gun_shot",          # 6 — постріл
    "jackhammer",        # 7 — відбійний молоток
    "siren",             # 8 — сирена
    "street_music",      # 9 — вулична музика
]

# Українські назви класів — для рисунків і таблиць у звіті
US8K_CLASSES_UA = {
    "air_conditioner": "кондиціонер",
    "car_horn": "автомобільний сигнал",
    "children_playing": "дитячі ігри",
    "dog_bark": "гавкіт собаки",
    "drilling": "дриль",
    "engine_idling": "двигун на холостому ходу",
    "gun_shot": "постріл",
    "jackhammer": "відбійний молоток",
    "siren": "сирена",
    "street_music": "вулична музика",
}

# Розбиття за fold-ами (рекомендація авторів UrbanSound8K: не перемішувати
# кліпи випадково, бо фрагменти одного запису лежать в одному fold-і)
US8K_TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7]
US8K_VAL_FOLDS = [8]
US8K_TEST_FOLDS = [9, 10]

# --- GTZAN ---
GTZAN_SEGMENT_SECONDS = 3.0  # кожен 30-секундний трек ріжемо на фрагменти по 3 с
GTZAN_SEGMENTS_PER_TRACK = 10
GTZAN_N_MELS = 96            # для музики беремо більше mel-смуг — тембр важливіший
GTZAN_SEGMENT_SAMPLES = int(SAMPLE_RATE * GTZAN_SEGMENT_SECONDS)  # 66 150 відліків
GTZAN_FRAMES = 1 + GTZAN_SEGMENT_SAMPLES // HOP_LENGTH            # 130 кадрів

GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

# Частки треків для розбиття GTZAN (за треками, стратифіковано за жанром)
GTZAN_TEST_FRACTION = 0.2  # 200 треків на тест
GTZAN_VAL_FRACTION = 0.1   # ~80 треків із решти — на валідацію

# ---------------------------------------------------------------------------
# Загальні налаштування навчання
# ---------------------------------------------------------------------------

SEED = 42  # фіксуємо генератори випадкових чисел для відтворюваності


def get_device() -> torch.device:
    """Повертає найшвидший доступний пристрій для обчислень.

    MPS (Metal Performance Shaders) — GPU на комп'ютерах Apple Silicon,
    CUDA — GPU NVIDIA, інакше — центральний процесор.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dirs() -> None:
    """Створює всі робочі теки, якщо їх ще немає."""
    for d in (RAW_DIR, FEATURES_DIR, MY_SOUNDS_DIR, RUNS_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
