"""
Підготовка колекції музичних жанрів GTZAN (проєкт Marsyas, Джордж Цанетакіс).

GTZAN — 1000 тридцятисекундних фрагментів (22 050 Гц, моно, WAV), по 100 на
кожен з 10 жанрів. Офіційна копія від проєкту Marsyas лежить на Hugging Face
(marsyas/gtzan) як один архів genres.tar.gz (~1,2 ГБ).

Обробка:
  * архів читається потоково, без розпакування на диск;
  * кожен трек доповнюється/обрізається до 30 с і ріжеться на 10 фрагментів
    по 3 с — так з 1000 треків отримуємо ~10 000 навчальних прикладів;
  * для кожного фрагмента рахується log-mel спектрограма 96 × 130.
Відомо, що файл jazz.00054.wav в GTZAN пошкоджений і не декодується — такі
треки пропускаються (тому в результаті 999 треків).

Результат: data/features/gtzan_logmel.npz з масивами
  X      — спектрограми фрагментів, [N, 96, 130], float16;
  y      — номер жанру фрагмента 0…9;
  track  — номер треку (0…998), до якого належить фрагмент;
  names  — імена файлів треків (індексуються номером треку).

Запуск:  python src/prepare_gtzan.py
"""

import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from huggingface_hub import hf_hub_download

from audio_features import fix_length, load_audio_from_bytes, log_mel_spectrogram
from config import (GTZAN_FEATURES, GTZAN_GENRES, GTZAN_N_MELS, GTZAN_SEGMENT_SAMPLES,
                    GTZAN_SEGMENTS_PER_TRACK, RAW_DIR, ensure_dirs)


def track_to_segments(audio_bytes: bytes):
    """Перетворює один трек на 10 спектрограм 3-секундних фрагментів.

    Повертає масив [10, 96, 130] або None, якщо файл не вдалося декодувати.
    """
    try:
        signal = load_audio_from_bytes(audio_bytes)
    except Exception:
        return None  # пошкоджений файл (jazz.00054) — пропускаємо
    total = GTZAN_SEGMENT_SAMPLES * GTZAN_SEGMENTS_PER_TRACK
    signal = fix_length(signal, total)  # деякі треки трохи коротші за 30 с
    segments = signal.reshape(GTZAN_SEGMENTS_PER_TRACK, GTZAN_SEGMENT_SAMPLES)
    return np.stack([log_mel_spectrogram(s, n_mels=GTZAN_N_MELS) for s in segments]).astype(np.float16)


def main() -> None:
    ensure_dirs()
    t0 = time.time()
    archive = hf_hub_download("marsyas/gtzan", "data/genres.tar.gz", repo_type="dataset",
                              local_dir=RAW_DIR)
    print(f"Архів завантажено за {time.time() - t0:.0f} с")

    # Потоково читаємо архів і збираємо байти WAV-файлів у пам'ять (~1,3 ГБ)
    names, genres, blobs = [], [], []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".wav"):
                continue
            fname = os.path.basename(member.name)       # напр. "blues.00000.wav"
            genre = fname.split(".")[0]
            names.append(fname)
            genres.append(GTZAN_GENRES.index(genre))
            blobs.append(tar.extractfile(member).read())

    # Сортуємо за іменем, щоб нумерація треків була стабільною
    order = np.argsort(names)
    names = [names[i] for i in order]
    genres = [genres[i] for i in order]
    blobs = [blobs[i] for i in order]
    print(f"Знайдено треків: {len(names)}")

    X, y, track, kept_names = [], [], [], []
    with ProcessPoolExecutor() as pool:
        for name, genre, segs in zip(names, genres, pool.map(track_to_segments, blobs, chunksize=4)):
            if segs is None:
                print(f"  пропущено пошкоджений файл: {name}")
                continue
            tid = len(kept_names)
            kept_names.append(name)
            X.append(segs)
            y.extend([genre] * len(segs))
            track.extend([tid] * len(segs))

    X = np.concatenate(X)
    np.savez(GTZAN_FEATURES, X=X, y=np.array(y, dtype=np.int64),
             track=np.array(track, dtype=np.int64), names=np.array(kept_names))
    os.remove(archive)  # звільняємо місце на диску
    print(f"Треків: {len(kept_names)}, фрагментів по 3 с: {X.shape}, {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
