"""
Пункт 4 завдання: перевірка навченої мережі на власних записах міських звуків.

Як користуватися:
  1. Запишіть телефоном 2–3 звуки класів, які є в UrbanSound8K (сирена,
     автомобільний сигнал, гавкіт собаки, вулична музика, дриль тощо).
  2. Збережіть файли у теку my_sounds/ у форматі WAV (моно бажано, тривалість
     від 4 с). Записи M4A/AMR/OGG з телефона варто конвертувати у WAV, напр.:
        ffmpeg -i запис.m4a -ac 1 -ar 22050 siren_my.wav
  3. Назвіть файл так, щоб він починався з назви класу, наприклад
     siren_street.wav або dog_bark_yard.wav — тоді скрипт сам порівняє
     прогноз зі справжнім класом.
  4. Запустіть:  python src/predict.py
     або для окремого файлу:  python src/predict.py шлях/до/файлу.wav

Як працює прогноз (функція predict_file):
  довгий запис ріжеться на вікна по 4 с з кроком 2 с (як кліпи UrbanSound8K),
  кожне вікно перетворюється на log-mel спектрограму і класифікується, а
  ймовірності всіх вікон усереднюються.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from audio_features import load_audio_file, log_mel_spectrogram, split_into_windows
from config import (FIGURES_DIR, MY_SOUNDS_DIR, RESULTS_DIR, RUNS_DIR, SAMPLE_RATE,
                    US8K_CLASSES, US8K_CLASSES_UA, US8K_CLIP_SAMPLES, US8K_N_MELS,
                    ensure_dirs, get_device)
from models import AudioCNN
from plots import plot_prediction

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


def load_model(device: torch.device, path: Path = RUNS_DIR / "us8k_best.pt"):
    """Завантажує найкращу модель UrbanSound8K разом з параметрами нормалізації."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = AudioCNN(len(ckpt["classes"]), cfg["base_channels"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["norm_mean"], ckpt["norm_std"]


@torch.no_grad()
def predict_file(path, model, mean: float, std: float, device: torch.device,
                 window_s: float = 4.0, hop_s: float = 2.0) -> dict:
    """Класифікує аудіофайл будь-якої тривалості.

    Повертає словник: тривалість, кількість вікон, усереднені ймовірності
    класів та повну спектрограму запису (для рисунка).
    """
    signal = load_audio_file(path)
    windows = split_into_windows(signal, int(window_s * SAMPLE_RATE), int(hop_s * SAMPLE_RATE))
    # Кожне вікно → спектрограма 64×173 → нормалізація тими самими mean/std, що в навчанні
    specs = np.stack([log_mel_spectrogram(w, n_mels=US8K_N_MELS) for w in windows])
    x = torch.from_numpy((specs - mean) / std).float().to(device)
    proba = torch.softmax(model(x), dim=1).cpu().numpy()
    return {
        "duration": len(signal) / SAMPLE_RATE,
        "n_windows": len(windows),
        "proba": proba.mean(axis=0),        # усереднення ймовірностей вікон
        "window_pred": proba.argmax(1),     # прогноз кожного окремого вікна
        "spec": log_mel_spectrogram(signal, n_mels=US8K_N_MELS),
    }


def expected_class(path: Path):
    """Визначає очікуваний клас з префікса імені файлу (якщо він є)."""
    stem = path.stem.lower()
    for c in sorted(US8K_CLASSES, key=len, reverse=True):
        if stem.startswith(c):
            return c
    return None


def main() -> None:
    ensure_dirs()
    device = get_device()
    model, mean, std = load_model(device)

    files = [Path(p) for p in sys.argv[1:]] or sorted(
        p for p in MY_SOUNDS_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
    if not files:
        print(f"У теці {MY_SOUNDS_DIR} немає аудіофайлів — додайте свої записи.")
        return

    # Необов'язковий файл з описом джерел записів (для звіту)
    sources_path = MY_SOUNDS_DIR / "sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else {}

    results = []
    for path in files:
        r = predict_file(path, model, mean, std, device)
        order = np.argsort(r["proba"])[::-1][:3]
        top3 = [(US8K_CLASSES[i], float(r["proba"][i])) for i in order]
        exp = expected_class(path)
        # Частка вікон, у яких окремо передбачено очікуваний клас
        window_hits = (float(np.mean(r["window_pred"] == US8K_CLASSES.index(exp)))
                       if exp else None)
        fig_name = f"my_{path.stem}.png"
        title = f"{path.name} ({r['duration']:.1f} с)"
        plot_prediction(r["spec"], [(US8K_CLASSES_UA[c], p) for c, p in top3], title,
                        FIGURES_DIR / fig_name)
        print(f"{path.name:28s} -> " + ", ".join(f"{c} {p:.0%}" for c, p in top3)
              + (f"   [очікувано: {exp}]" if exp else ""))
        results.append({
            "file": path.name, "duration": r["duration"], "n_windows": r["n_windows"],
            "expected": exp, "top3": top3, "correct": exp == top3[0][0] if exp else None,
            "window_hits": window_hits, "figure": fig_name,
            "source": sources.get(path.name, ""),
        })

    with open(RESULTS_DIR / "my_sounds_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
