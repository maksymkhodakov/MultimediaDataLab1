# Лабораторна робота №1
## "Методи обробки мультимедійної інформації"
## Ходаков Максим Олегович ШІ-2

**Тема:** класифікація звуків згортковими нейронними мережами — міські звуки
(UrbanSound8K) та музичні жанри (GTZAN).

### Структура

| Шлях | Призначення |
|---|---|
| `src/config.py` | шляхи, параметри спектрограм, розбиття даних |
| `src/audio_features.py` | завантаження звуку, log-mel спектрограма |
| `src/prepare_urbansound.py` | завантаження UrbanSound8K (Hugging Face) і обчислення спектрограм |
| `src/prepare_gtzan.py` | завантаження GTZAN (marsyas/gtzan), нарізання на фрагменти по 3 с |
| `src/models.py` | архітектура CNN `AudioCNN` |
| `src/training.py` | цикл навчання (AdamW + OneCycle), SpecAugment |
| `src/train_urbansound.py` | пункти 1–3: експерименти на UrbanSound8K |
| `src/predict.py` | пункт 4: прогноз для власних записів з теки `my_sounds/` |
| `src/train_gtzan.py` | пункт 5: класифікація жанрів GTZAN |
| `src/plots.py` | рисунки для звіту |
| `src/build_report.py`, `src/report_docx.py`, `src/report_text.py` | формування звіту за ДСТУ 3008:2015 |
| `results/` | метрики (JSON) і рисунки |
| `Звіт_Лабораторна_робота_1_Ходаков.docx` | готовий звіт |

### Запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/prepare_urbansound.py   # ~7 ГБ завантаження, обробляється шардами
python src/prepare_gtzan.py        # ~1,2 ГБ
python src/train_urbansound.py     # експерименти A–F
python src/predict.py              # власні записи з my_sounds/
python src/train_gtzan.py          # експерименти A–D
python src/build_report.py         # звіт .docx
```

### Власні записи (пункт 4)

Покладіть WAV-файли (моно, від 4 с) у `my_sounds/`, назвавши їх з префіксом класу
(`siren_…`, `dog_bark_…`, `car_horn_…`, `street_music_…` тощо), і запустіть
`python src/predict.py`, а потім `python src/build_report.py`. Запис з телефона
можна конвертувати так: `ffmpeg -i rec.m4a -ac 1 -ar 22050 siren_my.wav`.
Опис джерела запису можна додати в `my_sounds/sources.json`.
