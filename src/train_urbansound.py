"""
Пункти 1–3 завдання: CNN для класифікації міських звуків UrbanSound8K.

Що робить скрипт:
  1. завантажує готові log-mel спектрограми (див. prepare_urbansound.py);
  2. ділить дані за fold-ами: 1–7 — навчання, 8 — валідація, 9–10 — тест;
  3. нормалізує спектрограми (середнє та стандартне відхилення навчальної вибірки);
  4. навчає кілька варіантів мережі з різними параметрами (пункт 3 завдання);
  5. обирає найкращий варіант за точністю на валідації та оцінює його на тесті:
     accuracy, macro-F1, звіт по класах, матриця помилок;
  6. зберігає ваги найкращої моделі, метрики (JSON) і рисунки для звіту.

Запуск:  python src/train_urbansound.py
Повторний запуск пропускає вже навчені варіанти (їхні результати кешуються
у src/runs/us8k_<ключ>.pt); щоб перенавчити — видаліть відповідний файл.
"""

import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from config import (FIGURES_DIR, RESULTS_DIR, RUNS_DIR, SEED, US8K_CLASSES,
                    US8K_CLASSES_UA, US8K_FEATURES, US8K_N_MELS, US8K_FRAMES,
                    US8K_TEST_FOLDS, US8K_TRAIN_FOLDS, US8K_VAL_FOLDS,
                    ensure_dirs, get_device)
from models import AudioCNN, layer_table
from plots import (plot_confusion, plot_experiment_bars, plot_history,
                   plot_spectrogram_grid)
from training import Experiment, predict_proba, train_experiment

# ---------------------------------------------------------------------------
# Варіанти параметрів, які порівнюємо (пункт 3 завдання)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    Experiment("A", "базова (16-128 фільтрів, lr=2e-3)"),
    Experiment("B", "+ SpecAugment", spec_augment=True),
    Experiment("C", "+ SpecAugment, ширша (32-256)", spec_augment=True, base_channels=32),
    Experiment("D", "+ SpecAugment, lr=5e-4", spec_augment=True, lr=5e-4),
    Experiment("E", "ширша (32-256), без аугментації", base_channels=32),
    Experiment("F", "+ SpecAugment, ширша, 60 епох", spec_augment=True,
               base_channels=32, epochs=60),
]


def load_split(device: torch.device):
    """Завантажує спектрограми та ділить їх на train/val/test за fold-ами."""
    data = np.load(US8K_FEATURES)
    X, y, fold = data["X"].astype(np.float32), data["y"], data["fold"]

    masks = {
        "train": np.isin(fold, US8K_TRAIN_FOLDS),
        "val": np.isin(fold, US8K_VAL_FOLDS),
        "test": np.isin(fold, US8K_TEST_FOLDS),
    }
    # Параметри нормалізації рахуємо ЛИШЕ на навчальній вибірці,
    # щоб інформація з тесту не «просочувалась» у модель
    mean = float(X[masks["train"]].mean())
    std = float(X[masks["train"]].std())

    split = {}
    for name, m in masks.items():
        Xn = (X[m] - mean) / std
        split[name] = (torch.from_numpy(Xn).to(device), torch.from_numpy(y[m]).to(device))
    return X, y, fold, split, mean, std


def class_examples_figure(X: np.ndarray, y: np.ndarray) -> None:
    """Рисунок з прикладом спектрограми для кожного з 10 класів."""
    rng = np.random.default_rng(SEED)
    specs, titles = [], []
    for c, name in enumerate(US8K_CLASSES):
        idx = rng.choice(np.where(y == c)[0])
        specs.append(X[idx])
        titles.append(f"{c}. {US8K_CLASSES_UA[name]}")
    plot_spectrogram_grid(specs, titles, FIGURES_DIR / "us8k_class_examples.png")


def run_or_load(exp: Experiment, split: dict, device: torch.device, mean: float, std: float) -> dict:
    """Навчає варіант exp або завантажує його збережений результат."""
    ckpt_path = RUNS_DIR / f"us8k_{exp.key}.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt["config"] == exp.__dict__:
            print(f"  [{exp.key}] знайдено збережений результат, пропускаємо навчання")
            model = AudioCNN(10, exp.base_channels, exp.dropout).to(device)
            model.load_state_dict(ckpt["state_dict"])
            ckpt["model"] = model
            return ckpt

    print(f"  [{exp.key}] навчання: {exp.title}")
    res = train_experiment(exp, *split["train"], *split["val"], n_classes=10, device=device, seed=SEED)
    torch.save({
        "state_dict": res["model"].state_dict(),
        "config": res["config"], "history": res["history"],
        "best_val_acc": res["best_val_acc"], "best_epoch": res["best_epoch"],
        "n_params": res["n_params"], "train_seconds": res["train_seconds"],
        # Параметри нормалізації потрібні, щоб так само обробляти нові записи
        "norm_mean": mean, "norm_std": std, "classes": US8K_CLASSES,
    }, ckpt_path)
    return res


def main() -> None:
    ensure_dirs()
    device = get_device()
    print(f"Пристрій для навчання: {device}")

    X_raw, y_all, fold_all, split, mean, std = load_split(device)
    print(f"Спектрограми: {X_raw.shape} (кліпи × mel-смуги × часові кадри)")
    print("train:", len(split["train"][1]), "val:", len(split["val"][1]), "test:", len(split["test"][1]))
    class_counts = {name: int((y_all == c).sum()) for c, name in enumerate(US8K_CLASSES)}
    fold_counts = {int(f): int((fold_all == f).sum()) for f in range(1, 11)}
    class_examples_figure(X_raw, y_all)
    del X_raw

    X_test, y_test = split["test"]
    y_test_np = y_test.cpu().numpy()

    results = {}
    for exp in EXPERIMENTS:
        res = run_or_load(exp, split, device, mean, std)
        proba = predict_proba(res["model"], X_test)
        pred = proba.argmax(1)
        res["test_acc"] = accuracy_score(y_test_np, pred)
        res["macro_f1"] = f1_score(y_test_np, pred, average="macro")
        res["test_pred"] = pred
        res["title"] = exp.title
        results[exp.key] = res
        print(f"  {exp.key}. {exp.title:40s} test acc={res['test_acc']:.3f}  "
              f"macro-F1={res['macro_f1']:.3f}  val acc={res['best_val_acc']:.3f}  "
              f"({res['train_seconds']:.0f} с)")

    # Найкращий варіант обираємо за ВАЛІДАЦІЄЮ (fold 8), а не за тестом —
    # інакше оцінка на тесті була б оптимістично завищеною
    best_key = max(results, key=lambda k: results[k]["best_val_acc"])
    best = results[best_key]
    print(f"Найкращий за валідацією варіант: {best_key}. {best['title']} "
          f"(val acc = {best['best_val_acc']:.3f})")
    torch.save(torch.load(RUNS_DIR / f"us8k_{best_key}.pt", weights_only=False),
               RUNS_DIR / "us8k_best.pt")

    report = classification_report(y_test_np, best["test_pred"], target_names=US8K_CLASSES,
                                   digits=3, output_dict=True)
    print(classification_report(y_test_np, best["test_pred"], target_names=US8K_CLASSES, digits=3))

    # --- Рисунки ---
    ua_names = [US8K_CLASSES_UA[c] for c in US8K_CLASSES]
    plot_confusion(y_test_np, best["test_pred"], ua_names, FIGURES_DIR / "us8k_confusion.png")
    plot_history(results, FIGURES_DIR / "us8k_history_acc.png", metric="acc")
    plot_history(results, FIGURES_DIR / "us8k_history_loss.png", metric="loss")
    rows = [{"key": k, "title": r["title"], "test_acc": r["test_acc"], "macro_f1": r["macro_f1"]}
            for k, r in results.items()]
    plot_experiment_bars(rows, FIGURES_DIR / "us8k_experiments.png")

    # Найчастіші плутанини (пари «справжній → передбачений») для аналізу у звіті
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_test_np, best["test_pred"]):
        cm[t, p] += 1
    confusions = sorted(((int(cm[i, j]), US8K_CLASSES[i], US8K_CLASSES[j])
                         for i in range(10) for j in range(10) if i != j), reverse=True)[:8]

    # --- Збереження метрик у JSON (з нього будується звіт) ---
    base_model = AudioCNN(10, EXPERIMENTS[0].base_channels)
    summary = {
        "device": str(device),
        "n_clips": int(len(y_all)),
        "spec_shape": [US8K_N_MELS, US8K_FRAMES],
        "split_sizes": {k: int(len(v[1])) for k, v in split.items()},
        "class_counts": class_counts,
        "fold_counts": fold_counts,
        "norm": {"mean": mean, "std": std},
        "layers": [list(map(str, r[:3])) + [r[3]] for r in layer_table(base_model, (US8K_N_MELS, US8K_FRAMES))],
        "experiments": [{
            "key": k, "title": r["title"], "config": r["config"],
            "n_params": r["n_params"], "best_epoch": r["best_epoch"],
            "best_val_acc": r["best_val_acc"], "test_acc": r["test_acc"],
            "macro_f1": r["macro_f1"], "train_seconds": r["train_seconds"],
            "final_train_acc": r["history"]["train_acc"][-1],
        } for k, r in results.items()],
        "best_key": best_key,
        "report": report,
        "top_confusions": confusions,
    }
    with open(RESULTS_DIR / "us8k_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Результати збережено у {RESULTS_DIR / 'us8k_results.json'}")


if __name__ == "__main__":
    main()
