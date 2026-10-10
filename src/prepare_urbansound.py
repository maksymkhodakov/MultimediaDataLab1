"""
Підготовка датасету UrbanSound8K: завантаження та обчислення log-mel спектрограм.

Джерело — копія оригінального UrbanSound8K на Hugging Face Hub
(danavery/urbansound8K, ~7 ГБ у форматі Parquet, 16 файлів-шардів).
Аудіо там те саме, що в оригіналі; для кожного кліпу також збережено номер
fold (одна з 10 заздалегідь заданих частин датасету) та клас.

Щоб не тримати на диску всі 7 ГБ одночасно, шарди обробляються по одному:
  завантажити шард → порахувати спектрограми → зберегти проміжний результат →
  видалити шард. Проміжні результати дозволяють продовжити роботу після збою.

Результат: файл data/features/urbansound8k_logmel.npz з масивами
  X      — спектрограми, форма [8732, 64, 173], float16 (економимо пам'ять);
  y      — номери класів 0…9;
  fold   — номер fold-а 1…10;
  fsid   — ідентифікатор вихідного запису Freesound;
  names  — імена файлів кліпів.

Запуск:  python src/prepare_urbansound.py
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from audio_features import fix_length, load_audio_from_bytes, log_mel_spectrogram
from config import (FEATURES_DIR, RAW_DIR, US8K_CLIP_SAMPLES, US8K_FEATURES,
                    US8K_N_MELS, ensure_dirs)

REPO_ID = "danavery/urbansound8K"  # ідентифікатор датасету на Hugging Face
PARTS_DIR = FEATURES_DIR / "us8k_parts"  # проміжні результати по шардах


def clip_to_features(audio_bytes: bytes) -> np.ndarray:
    """Перетворює один кліп (байти WAV-файлу) на log-mel спектрограму 64×173.

    Кліпи UrbanSound8K мають тривалість до 4 с, тому всі вони доповнюються
    тишею до рівно 4 с — так ширина спектрограми однакова для всіх.
    """
    signal = load_audio_from_bytes(audio_bytes)
    signal = fix_length(signal, US8K_CLIP_SAMPLES)
    return log_mel_spectrogram(signal, n_mels=US8K_N_MELS).astype(np.float16)


def process_shard(filename: str, pool: ProcessPoolExecutor) -> None:
    """Завантажує один Parquet-шард, рахує спектрограми та видаляє шард."""
    part_path = PARTS_DIR / (os.path.basename(filename) + ".npz")
    if part_path.exists():
        print(f"  {filename}: вже оброблено, пропускаємо")
        return

    t0 = time.time()
    local = hf_hub_download(REPO_ID, filename, repo_type="dataset", local_dir=RAW_DIR)
    print(f"  {filename}: завантажено за {time.time() - t0:.0f} с")

    # Читаємо шард групами рядків (row groups), щоб не тримати в пам'яті
    # усі аудіобайти одночасно
    pf = pq.ParquetFile(local)
    feats, labels, folds, fsids, names = [], [], [], [], []
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=["audio", "slice_file_name", "fsID", "fold", "classID"])
        rows = table.to_pydict()
        audio_bytes = [a["bytes"] for a in rows["audio"]]
        # Паралельно на всіх ядрах процесора рахуємо спектрограми
        feats.extend(pool.map(clip_to_features, audio_bytes, chunksize=8))
        labels.extend(rows["classID"])
        folds.extend(rows["fold"])
        fsids.extend(rows["fsID"])
        names.extend(rows["slice_file_name"])

    np.savez(part_path, X=np.stack(feats), y=np.array(labels, dtype=np.int64),
             fold=np.array(folds, dtype=np.int64), fsid=np.array(fsids, dtype=np.int64),
             names=np.array(names))
    os.remove(local)  # звільняємо місце на диску — сирий шард більше не потрібен
    print(f"  {filename}: {len(labels)} кліпів, усього {time.time() - t0:.0f} с")


def main() -> None:
    ensure_dirs()
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Перелік Parquet-файлів датасету отримуємо через API Hugging Face
    files = sorted(f for f in HfApi().list_repo_files(REPO_ID, repo_type="dataset")
                   if f.endswith(".parquet"))
    print(f"Шардів UrbanSound8K: {len(files)}")

    with ProcessPoolExecutor() as pool:
        for f in files:
            process_shard(f, pool)

    # Об'єднуємо проміжні результати в один файл і впорядковуємо за іменем кліпу,
    # щоб порядок прикладів був однаковим при кожному запуску
    parts = [np.load(p) for p in sorted(PARTS_DIR.glob("*.npz"))]
    data = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "y", "fold", "fsid", "names")}
    order = np.argsort(data["names"])
    data = {k: v[order] for k, v in data.items()}
    np.savez(US8K_FEATURES, **data)
    print(f"Збережено {US8K_FEATURES}: X={data['X'].shape}")


if __name__ == "__main__":
    main()
