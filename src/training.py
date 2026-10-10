"""
Спільний код навчання та оцінювання CNN для обох частин роботи.

Що тут є:
  * Experiment       — опис одного варіанта гіперпараметрів;
  * spec_augment     — аугментація спектрограм SpecAugment;
  * train_experiment — повний цикл навчання з вибором найкращої епохи за валідацією;
  * predict_proba    — імовірності класів для набору спектрограм;
  * set_seed         — фіксація генераторів випадкових чисел.
"""

import copy
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import AudioCNN, count_parameters


@dataclass
class Experiment:
    """Опис одного експерименту (варіанта параметрів)."""
    key: str                    # коротка мітка: "A", "B", …
    title: str                  # опис українською для таблиць і графіків
    base_channels: int = 16     # фільтрів у першому блоці (16 → 16-32-64-128)
    lr: float = 2e-3            # максимальна швидкість навчання для OneCycle
    epochs: int = 30            # кількість епох
    batch_size: int = 64        # розмір міні-батчу
    weight_decay: float = 1e-2  # коефіцієнт L2-регуляризації ваг в AdamW
    dropout: float = 0.3        # частка Dropout перед останнім шаром
    spec_augment: bool = False  # чи вмикати SpecAugment
    freq_mask: int = 8          # макс. ширина маски частот (у mel-смугах)
    time_mask: int = 20         # макс. ширина маски часу (у кадрах)
    n_masks: int = 2            # скільки масок кожного типу накладати


def set_seed(seed: int) -> None:
    """Фіксує всі генератори випадкових чисел, щоб результати повторювались."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def spec_augment(x: torch.Tensor, freq_mask: int, time_mask: int, n_masks: int) -> torch.Tensor:
    """SpecAugment — випадкове «затирання» смуг частот і проміжків часу.

    Для кожного прикладу в батчі n_masks разів вибирається випадкова горизонтальна
    смуга (діапазон mel-смуг шириною до freq_mask) і вертикальна смуга (проміжок
    до time_mask кадрів), які заповнюються нулем — середнім значенням після
    нормалізації. Так мережа не може покладатися на одну окрему деталь
    спектрограми і змушена використовувати всю картину — це зменшує перенавчання.

    Реалізовано векторно (масками-порівняннями), щоб усе рахувалося на GPU.
    """
    b, n_mels, n_frames = x.shape
    device = x.device
    mel_idx = torch.arange(n_mels, device=device).view(1, n_mels, 1)
    frame_idx = torch.arange(n_frames, device=device).view(1, 1, n_frames)
    keep = torch.ones_like(x, dtype=torch.bool)
    for _ in range(n_masks):
        # Маска частот: [f0, f0 + f)
        f = torch.randint(0, freq_mask + 1, (b, 1, 1), device=device)
        f0 = (torch.rand(b, 1, 1, device=device) * (n_mels - f)).long()
        keep &= ~((mel_idx >= f0) & (mel_idx < f0 + f))
        # Маска часу: [t0, t0 + t)
        t = torch.randint(0, time_mask + 1, (b, 1, 1), device=device)
        t0 = (torch.rand(b, 1, 1, device=device) * (n_frames - t)).long()
        keep &= ~((frame_idx >= t0) & (frame_idx < t0 + t))
    return x * keep


@torch.no_grad()
def predict_proba(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    """Повертає матрицю ймовірностей [кількість прикладів × класи].

    softmax перетворює «сирі» оцінки мережі (logits) на ймовірності,
    що в сумі дають 1.
    """
    model.eval()  # вимикаємо Dropout і переводимо BatchNorm у режим інференсу
    out = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        out.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(out).numpy()


def train_experiment(exp: Experiment, X_train: torch.Tensor, y_train: torch.Tensor,
                     X_val: torch.Tensor, y_val: torch.Tensor, n_classes: int,
                     device: torch.device, seed: int = 42) -> dict:
    """Навчає одну модель за параметрами exp.

    Алгоритм:
      * оптимізатор AdamW — Adam з окремим (decoupled) weight decay;
      * розклад швидкості навчання OneCycle: перші 30 % кроків lr зростає до
        exp.lr, потім плавно (косинусом) спадає майже до нуля — це стабільніше
        за сталу швидкість і дає кращу збіжність;
      * функція втрат — перехресна ентропія (cross-entropy);
      * після кожної епохи міряємо точність на валідації і запам'ятовуємо
        ваги найкращої епохи (рання зупинка «в ретроспективі»).

    Усі тензори вже мають бути на пристрої device.
    Повертає словник з навченою моделлю (найкращі ваги), історією навчання,
    кількістю параметрів і часом навчання.
    """
    set_seed(seed)
    model = AudioCNN(n_classes=n_classes, base_channels=exp.base_channels,
                     dropout=exp.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp.lr, weight_decay=exp.weight_decay)
    steps_per_epoch = (len(X_train) + exp.batch_size - 1) // exp.batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=exp.lr, epochs=exp.epochs, steps_per_epoch=steps_per_epoch)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc, best_state, best_epoch = -1.0, None, 0
    generator = torch.Generator().manual_seed(seed)
    t_start = time.time()

    for epoch in range(1, exp.epochs + 1):
        model.train()  # вмикаємо Dropout і навчальний режим BatchNorm
        # Перемішуємо навчальні приклади на кожній епосі
        perm = torch.randperm(len(X_train), generator=generator).to(device)
        total_loss, total_correct = 0.0, 0
        for i in range(0, len(X_train), exp.batch_size):
            idx = perm[i:i + exp.batch_size]
            xb, yb = X_train[idx], y_train[idx]
            if exp.spec_augment:
                xb = spec_augment(xb, exp.freq_mask, exp.time_mask, exp.n_masks)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)  # обнуляємо старі градієнти
            loss.backward()                        # зворотне поширення помилки
            optimizer.step()                       # крок оптимізатора
            scheduler.step()                       # оновлюємо lr за розкладом OneCycle

            total_loss += loss.item() * len(yb)
            total_correct += (logits.argmax(1) == yb).sum().item()

        # Оцінка на валідаційній вибірці
        model.eval()
        with torch.no_grad():
            val_logits = torch.cat([model(X_val[j:j + 256]) for j in range(0, len(X_val), 256)])
            val_loss = criterion(val_logits, y_val).item()
            val_acc = (val_logits.argmax(1) == y_val).float().mean().item()

        history["train_loss"].append(total_loss / len(X_train))
        history["train_acc"].append(total_correct / len(X_train))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(scheduler.get_last_lr()[0])

        # Запам'ятовуємо копію ваг найкращої за валідацією епохи
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())

        print(f"    [{exp.key}] епоха {epoch:2d}/{exp.epochs}: "
              f"train loss={history['train_loss'][-1]:.3f} acc={history['train_acc'][-1]:.3f} | "
              f"val loss={val_loss:.3f} acc={val_acc:.3f}", flush=True)

    model.load_state_dict(best_state)  # повертаємо найкращі ваги
    return {
        "model": model,
        "history": history,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "n_params": count_parameters(model),
        "train_seconds": time.time() - t_start,
        "config": asdict(exp),
    }
