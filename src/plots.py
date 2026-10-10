"""
Побудова рисунків для звіту: спектрограми, криві навчання, матриці помилок,
порівняння експериментів.

Усі рисунки зберігаються у PNG з роздільністю 200 dpi у теку results/figures.
Підписи — українською, бо рисунки вставляються у звіт.
"""

import matplotlib

matplotlib.use("Agg")  # малюємо без вікон — лише у файли

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from config import HOP_LENGTH, SAMPLE_RATE

# Узгоджені кольори для всіх рисунків: навчання — синій, валідація — помаранчевий
COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"
COLOR_BAR = "#2a78d6"
TEXT_SECONDARY = "#52514e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # шрифт із підтримкою кирилиці
    "font.size": 10,
    "axes.spines.top": False,       # прибираємо зайві рамки
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e5e5e3",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8984",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.axisbelow": True,         # сітка — під стовпчиками та лініями, а не поверх
})


def plot_spectrogram_grid(specs: list, titles: list, path, n_cols: int = 5,
                          seconds: float = None) -> None:
    """Сітка log-mel спектрограм (по одній на клас) — щоб показати їхню «текстуру»."""
    n_rows = int(np.ceil(len(specs) / n_cols))
    # constrained_layout автоматично розсуває підграфіки, щоб підписи не накладались
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.9 * n_rows),
                             layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    for ax, spec, title in zip(axes, specs, titles):
        dur = seconds if seconds else spec.shape[1] * HOP_LENGTH / SAMPLE_RATE
        im = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma",
                       extent=[0, dur, 0, spec.shape[0]], vmin=-80, vmax=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("час, с", fontsize=8)
        ax.set_ylabel("mel-смуга", fontsize=8)
        ax.grid(False)
        ax.tick_params(labelsize=7)
    for ax in axes[len(specs):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="дБ")
    fig.savefig(path)
    plt.close(fig)


def plot_history(results: dict, path, metric: str = "acc") -> None:
    """Криві навчання для кількох експериментів: точність або втрати по епохах.

    results — словник {ключ: {"title": ..., "history": {...}}}.
    Кожен експеримент — окремий підграфік (small multiples), щоб криві не
    накладалися; суцільна лінія — навчання, пунктир — валідація.
    """
    n = len(results)
    n_cols = min(n, 2)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 3.4 * n_rows),
                             sharey=True, squeeze=False)
    axes = axes.ravel()
    label = "точність" if metric == "acc" else "функція втрат"
    for ax, (key, r) in zip(axes, results.items()):
        h = r["history"]
        epochs = np.arange(1, len(h[f"train_{metric}"]) + 1)
        ax.plot(epochs, h[f"train_{metric}"], color=COLOR_TRAIN, lw=2, label="навчання")
        ax.plot(epochs, h[f"val_{metric}"], color=COLOR_VAL, lw=2, ls="--", label="валідація")
        # Позначаємо найкращу за валідацією епоху
        be = r["best_epoch"]
        ax.plot(be, h[f"val_{metric}"][be - 1], "o", color=COLOR_VAL, ms=8,
                markeredgecolor="white", markeredgewidth=2)
        ax.set_title(f"{key}. {r['title']}", fontsize=10)
        ax.set_xlabel("епоха")
        ax.set_ylabel(label)
        ax.legend(loc="lower right" if metric == "acc" else "upper right", frameon=False)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(y_true, y_pred, class_names: list, path, title: str = "") -> None:
    """Матриця помилок, нормована по рядках (частка кожного справжнього класу).

    Рядок — справжній клас, стовпець — передбачений. Діагональ — правильні
    відповіді; великі значення поза діагоналлю показують, які класи плутаються.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("передбачений клас")
    ax.set_ylabel("справжній клас")
    if title:
        ax.set_title(title)
    # Підписуємо лише ненульові клітинки, щоб матриця не була перевантажена
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] > 0:
                color = "white" if cm_norm[i, j] > 0.55 else "#0b0b0b"
                ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", ha="center",
                        va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="частка класу")
    fig.savefig(path)
    plt.close(fig)


def plot_experiment_bars(rows: list, path, metrics=(("test_acc", "accuracy"), ("macro_f1", "macro-F1"))) -> None:
    """Порівняння експериментів: горизонтальні стовпчики для кожної метрики.

    rows — список словників з полями key, title та значеннями метрик.
    Кожна метрика — окремий підграфік (одна вісь на графік).
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 0.6 * len(rows) + 1.2),
                             sharey=True, squeeze=False)
    labels = [f"{r['key']}. {r['title']}" for r in rows]
    y = np.arange(len(rows))[::-1]
    for ax, (field, name) in zip(axes.ravel(), metrics):
        vals = [r[field] for r in rows]
        ax.barh(y, vals, color=COLOR_BAR, height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=9, color="#0b0b0b")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(min(vals) - 0.1, 1.0)
        ax.set_xlabel(name)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_prediction(spec: np.ndarray, top: list, title: str, path) -> None:
    """Спектрограма власного запису + стовпчики top-3 прогнозу мережі."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.0),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    dur = spec.shape[1] * HOP_LENGTH / SAMPLE_RATE
    ax1.imshow(spec, origin="lower", aspect="auto", cmap="magma",
               extent=[0, dur, 0, spec.shape[0]], vmin=-80, vmax=0)
    ax1.set_title(title, fontsize=10)
    ax1.set_xlabel("час, с")
    ax1.set_ylabel("mel-смуга")
    ax1.grid(False)
    names = [t[0] for t in top][::-1]
    probs = [t[1] for t in top][::-1]
    ax2.barh(range(len(top)), probs, color=COLOR_BAR, height=0.6)
    for i, p in enumerate(probs):
        ax2.text(p + 0.02, i, f"{p:.0%}", va="center", fontsize=9)
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels(names)
    ax2.set_xlim(0, 1.15)
    ax2.set_xlabel("імовірність")
    ax2.set_title("top-3 прогноз", fontsize=10)
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
