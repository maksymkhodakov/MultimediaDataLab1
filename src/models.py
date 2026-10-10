"""
Архітектура згорткової нейронної мережі (CNN) для класифікації спектрограм.

Мережа складається з 4 згорткових блоків:
    згортка 3×3 → BatchNorm → ReLU → MaxPool 2×2
після яких іде глобальне усереднення (Global Average Pooling), Dropout
і повнозв'язний шар на кількість класів.

Кожен MaxPool зменшує висоту й ширину «картинки» вдвічі, а кількість фільтрів
подвоюється: перші шари бачать дрібні локальні візерунки (короткі імпульси,
гармоніки), глибші — складніші та довші структури (ритм, дуги сирени).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Один згортковий блок: Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            # Згортка 3×3 з доповненням 1 зберігає розмір карти ознак;
            # bias=True залишаємо для відповідності класичній схемі
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # Batch Normalization — нормалізація виходу по міні-батчу:
            # пришвидшує та стабілізує навчання
            nn.BatchNorm2d(out_channels),
            # ReLU (Rectified Linear Unit) — нелінійність max(0, x)
            nn.ReLU(inplace=True),
            # Максимум у вікні 2×2 — зменшує розмір удвічі та дає
            # невелику стійкість до зсувів у часі й частоті
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AudioCNN(nn.Module):
    """Згорткова мережа для класифікації log-mel спектрограм.

    Параметри:
        n_classes     — кількість класів на виході (10 для обох датасетів);
        base_channels — кількість фільтрів у першому блоці; у наступних блоках
                        вона подвоюється: 16-32-64-128 (базова) або
                        32-64-128-256 (ширша мережа);
        dropout       — частка нейронів, що випадково «вимикаються» під час
                        навчання перед останнім шаром (захист від перенавчання).
    """

    def __init__(self, n_classes: int = 10, base_channels: int = 16, dropout: float = 0.3):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            ConvBlock(1, c),          # вхід — 1 канал (одна спектрограма)
            ConvBlock(c, 2 * c),
            ConvBlock(2 * c, 4 * c),
            ConvBlock(4 * c, 8 * c),
        )
        # Global Average Pooling: усереднюємо кожну карту ознак по частоті та
        # часу → вектор довжини 8c. Завдяки цьому мережа працює зі
        # спектрограмою будь-якої ширини і має мало параметрів
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(8 * c, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x має форму [батч, mel-смуги, кадри]; додаємо вимір каналу → [B, 1, M, T]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.pool(x).flatten(1)  # [B, 8c]
        x = self.dropout(x)
        return self.classifier(x)    # «сирі» оцінки (logits) для кожного класу


def count_parameters(model: nn.Module) -> int:
    """Кількість параметрів моделі, що навчаються."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_table(model: AudioCNN, input_shape: tuple) -> list:
    """Будує таблицю шарів (назва, форма виходу, кількість параметрів).

    Пропускає через мережу один «порожній» приклад і за допомогою хуків
    (forward hooks) запам'ятовує форму виходу кожного шару. Таблиця
    використовується у звіті.
    """
    rows = []
    hooks = []

    def make_hook(name, module):
        def hook(_m, _inp, out):
            params = sum(p.numel() for p in module.parameters(recurse=False))
            rows.append((name, type(module).__name__, tuple(out.shape[1:]), params))
        return hook

    # Реєструємо хуки лише на «листових» шарах (без вкладених модулів)
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name, module)))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape))
    for h in hooks:
        h.remove()
    return rows
