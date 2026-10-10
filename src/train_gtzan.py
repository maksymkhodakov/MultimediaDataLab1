"""
Пункт 5 завдання: CNN для класифікації музичних жанрів (колекція GTZAN).

Підхід:
  * кожен 30-секундний трек розрізаний на 10 фрагментів по 3 с (prepare_gtzan.py);
  * дані ділимо на train/val/test ЗА ТРЕКАМИ, стратифіковано за жанром
    (20 % треків — тест, з решти ~10 % — валідація), щоб фрагменти одного треку
    не потрапили одночасно в навчання й тест — інакше точність була б завищена;
  * мережа класифікує кожен фрагмент окремо (точність «на рівні фрагментів»);
  * прогноз для треку — усереднення ймовірностей усіх його 10 фрагментів
    («голосування», точність «на рівні треків»).

Запуск:  python src/train_gtzan.py
"""

import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from config import (FIGURES_DIR, GTZAN_FEATURES, GTZAN_FRAMES, GTZAN_GENRES,
                    GTZAN_N_MELS, GTZAN_TEST_FRACTION, GTZAN_VAL_FRACTION,
                    RESULTS_DIR, RUNS_DIR, SEED, ensure_dirs, get_device)
from models import AudioCNN, layer_table
from plots import (plot_confusion, plot_experiment_bars, plot_history,
                   plot_spectrogram_grid)
from training import Experiment, predict_proba, train_experiment

# Для 96 mel-смуг маска частот трохи ширша, ніж для UrbanSound8K
EXPERIMENTS = [
    Experiment("A", "базова (16-128), без аугментації"),
    Experiment("B", "ширша (32-256) + SpecAugment", base_channels=32,
               spec_augment=True, freq_mask=12, time_mask=15),
    Experiment("C", "базова (16-128) + SpecAugment", spec_augment=True,
               freq_mask=12, time_mask=15),
    Experiment("D", "ширша + SpecAugment, 50 епох", base_channels=32,
               spec_augment=True, freq_mask=12, time_mask=15, epochs=50),
]


def split_tracks(track_genre: np.ndarray):
    """Стратифіковане розбиття номерів треків на train/val/test."""
    track_ids = np.arange(len(track_genre))
    trainval, test = train_test_split(track_ids, test_size=GTZAN_TEST_FRACTION,
                                      stratify=track_genre, random_state=SEED)
    train, val = train_test_split(trainval, test_size=GTZAN_VAL_FRACTION,
                                  stratify=track_genre[trainval], random_state=SEED)
    return np.sort(train), np.sort(val), np.sort(test)


def track_vote(proba: np.ndarray, tracks: np.ndarray):
    """Усереднює ймовірності фрагментів кожного треку («голосування»).

    Повертає (номери треків, усереднені ймовірності [треки × жанри]).
    """
    uniq = np.unique(tracks)
    mean_proba = np.stack([proba[tracks == t].mean(axis=0) for t in uniq])
    return uniq, mean_proba


def run_or_load(exp, data, device, mean, std):
    """Навчає варіант exp або завантажує збережений результат з диска."""
    ckpt_path = RUNS_DIR / f"gtzan_{exp.key}.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt["config"] == exp.__dict__:
            print(f"  [{exp.key}] знайдено збережений результат, пропускаємо навчання")
            model = AudioCNN(10, exp.base_channels, exp.dropout).to(device)
            model.load_state_dict(ckpt["state_dict"])
            ckpt["model"] = model
            return ckpt
    print(f"  [{exp.key}] навчання: {exp.title}")
    res = train_experiment(exp, *data["train"], *data["val"], n_classes=10, device=device, seed=SEED)
    torch.save({
        "state_dict": res["model"].state_dict(), "config": res["config"],
        "history": res["history"], "best_val_acc": res["best_val_acc"],
        "best_epoch": res["best_epoch"], "n_params": res["n_params"],
        "train_seconds": res["train_seconds"], "norm_mean": mean, "norm_std": std,
        "classes": GTZAN_GENRES,
    }, ckpt_path)
    return res


def main() -> None:
    ensure_dirs()
    device = get_device()
    print(f"Пристрій для навчання: {device}")

    npz = np.load(GTZAN_FEATURES)
    X, y, track = npz["X"].astype(np.float32), npz["y"], npz["track"]
    n_tracks = int(track.max()) + 1
    track_genre = np.array([y[track == t][0] for t in range(n_tracks)])
    print(f"Треків: {n_tracks}, фрагментів по 3 с: {X.shape}")

    tr_ids, va_ids, te_ids = split_tracks(track_genre)
    print(f"Треки train/val/test: {len(tr_ids)}/{len(va_ids)}/{len(te_ids)}")
    masks = {"train": np.isin(track, tr_ids), "val": np.isin(track, va_ids), "test": np.isin(track, te_ids)}

    # Рисунок: приклад фрагмента для кожного жанру
    rng = np.random.default_rng(SEED)
    idx = [rng.choice(np.where(y == g)[0]) for g in range(10)]
    plot_spectrogram_grid([X[i] for i in idx], GTZAN_GENRES, FIGURES_DIR / "gtzan_genre_examples.png")

    # Нормалізація за статистикою навчальної вибірки
    mean, std = float(X[masks["train"]].mean()), float(X[masks["train"]].std())
    data = {k: (torch.from_numpy((X[m] - mean) / std).to(device), torch.from_numpy(y[m]).to(device))
            for k, m in masks.items()}
    del X

    y_test_seg = y[masks["test"]]
    test_tracks = track[masks["test"]]

    results = {}
    for exp in EXPERIMENTS:
        res = run_or_load(exp, data, device, mean, std)
        proba = predict_proba(res["model"], data["test"][0])
        uniq, track_proba = track_vote(proba, test_tracks)
        res["seg_acc"] = accuracy_score(y_test_seg, proba.argmax(1))
        res["track_true"] = track_genre[uniq]
        res["track_pred"] = track_proba.argmax(1)
        res["track_acc"] = accuracy_score(res["track_true"], res["track_pred"])
        res["track_f1"] = f1_score(res["track_true"], res["track_pred"], average="macro")
        res["title"] = exp.title
        results[exp.key] = res
        print(f"  {exp.key}. {exp.title:38s} фрагменти: {res['seg_acc']:.3f}  "
              f"треки (голосування): {res['track_acc']:.3f}  val={res['best_val_acc']:.3f}  "
              f"({res['train_seconds']:.0f} с)")

    best_key = max(results, key=lambda k: results[k]["best_val_acc"])
    best = results[best_key]
    print(f"Найкращий за валідацією варіант: {best_key}. {best['title']} "
          f"(val acc фрагментів = {best['best_val_acc']:.3f})")
    torch.save(torch.load(RUNS_DIR / f"gtzan_{best_key}.pt", weights_only=False), RUNS_DIR / "gtzan_best.pt")

    report = classification_report(best["track_true"], best["track_pred"], target_names=GTZAN_GENRES,
                                   digits=3, output_dict=True)
    print(classification_report(best["track_true"], best["track_pred"], target_names=GTZAN_GENRES, digits=3))

    plot_confusion(best["track_true"], best["track_pred"], GTZAN_GENRES, FIGURES_DIR / "gtzan_confusion.png")
    plot_history(results, FIGURES_DIR / "gtzan_history_acc.png", metric="acc")
    plot_history(results, FIGURES_DIR / "gtzan_history_loss.png", metric="loss")
    rows = [{"key": k, "title": r["title"], "seg_acc": r["seg_acc"], "track_acc": r["track_acc"]}
            for k, r in results.items()]
    plot_experiment_bars(rows, FIGURES_DIR / "gtzan_experiments.png",
                         metrics=(("seg_acc", "точність на фрагментах"),
                                  ("track_acc", "точність на треках (голосування)")))

    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(best["track_true"], best["track_pred"]):
        cm[t, p] += 1
    confusions = sorted(((int(cm[i, j]), GTZAN_GENRES[i], GTZAN_GENRES[j])
                         for i in range(10) for j in range(10) if i != j and cm[i, j] > 0), reverse=True)

    summary = {
        "n_tracks": n_tracks,
        "n_segments": int(len(y)),
        "spec_shape": [GTZAN_N_MELS, GTZAN_FRAMES],
        "split_tracks": {"train": len(tr_ids), "val": len(va_ids), "test": len(te_ids)},
        "split_segments": {k: int(m.sum()) for k, m in masks.items()},
        "layers": [list(map(str, r[:3])) + [r[3]] for r in
                   layer_table(AudioCNN(10, 32), (GTZAN_N_MELS, GTZAN_FRAMES))],
        "experiments": [{
            "key": k, "title": r["title"], "config": r["config"], "n_params": r["n_params"],
            "best_epoch": r["best_epoch"], "best_val_acc": r["best_val_acc"],
            "seg_acc": r["seg_acc"], "track_acc": r["track_acc"], "track_f1": r["track_f1"],
            "train_seconds": r["train_seconds"], "final_train_acc": r["history"]["train_acc"][-1],
        } for k, r in results.items()],
        "best_key": best_key,
        "report": report,
        "top_confusions": confusions,
    }
    with open(RESULTS_DIR / "gtzan_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Результати збережено у {RESULTS_DIR / 'gtzan_results.json'}")


if __name__ == "__main__":
    main()
