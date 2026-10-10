"""
Перетворення звуку на log-mel спектрограму.

Ідея: звуковий сигнал (одновимірний масив відліків) перетворюємо на «картинку»
розміром [кількість mel-смуг × кількість часових кадрів]:
  * по горизонталі — час;
  * по вертикалі — частота в mel-шкалі (наближена до сприйняття людським вухом:
    низькі частоти розрізняємо детальніше, ніж високі);
  * яскравість пікселя — енергія в децибелах.
Із такою «картинкою» далі працює звичайна згорткова нейронна мережа (CNN).
"""

import io

import librosa
import numpy as np
import soundfile as sf

from config import HOP_LENGTH, N_FFT, SAMPLE_RATE


def load_audio_from_bytes(data: bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Декодує аудіофайл (WAV/FLAC/OGG…), записаний у пам'яті як bytes.

    Повертає моно-сигнал з частотою дискретизації sr у форматі float32.
    """
    # soundfile читає байти напряму, без збереження на диск;
    # always_2d=True гарантує форму [відліки, канали] навіть для моно
    signal, orig_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    signal = signal.mean(axis=1)  # стерео → моно: усереднюємо канали
    if orig_sr != sr:
        # Передискретизація до спільної частоти, щоб усі спектрограми були сумісні
        signal = librosa.resample(signal, orig_sr=orig_sr, target_sr=sr)
    return signal.astype(np.float32)


def load_audio_file(path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Завантажує аудіофайл з диска як моно-сигнал з частотою sr.

    librosa.load підтримує WAV, FLAC, OGG, MP3 (через soundfile/audioread),
    тому сюди можна подавати й записи з телефона, конвертовані у WAV.
    """
    signal, _ = librosa.load(str(path), sr=sr, mono=True)
    return signal.astype(np.float32)


def fix_length(signal: np.ndarray, n_samples: int) -> np.ndarray:
    """Приводить сигнал до рівно n_samples відліків.

    Коротші записи доповнюються нулями (тишею) в кінці, довші — обрізаються.
    Це потрібно, щоб усі спектрограми мали однакову ширину.
    """
    if len(signal) >= n_samples:
        return signal[:n_samples]
    return np.pad(signal, (0, n_samples - len(signal)))


def log_mel_spectrogram(signal: np.ndarray, n_mels: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Обчислює log-mel спектрограму сигналу.

    Кроки:
      1. STFT (коротке перетворення Фур'є): сигнал ділиться на кадри по N_FFT
         відліків з кроком HOP_LENGTH, для кожного кадру рахується спектр;
      2. квадрат модуля спектра — потужність кожної частоти;
      3. множення на банк трикутних mel-фільтрів — групуємо частоти у n_mels смуг;
      4. переведення потужності в децибели: 10·log10(P / P_max), обмежуючи
         динамічний діапазон 80 дБ (усе тихіше за −80 дБ вважаємо −80).

    Повертає масив float32 форми [n_mels, кадри] зі значеннями від −80 до 0 дБ.
    """
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=n_mels,
        power=2.0,  # спектр потужності
    )
    # ref=np.max — нормуємо на максимум кліпу, тож гучність запису (відстань
    # до мікрофона) менше впливає на результат; top_db — нижня межа діапазону
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    return log_mel.astype(np.float32)


def split_into_windows(signal: np.ndarray, window: int, hop: int) -> list:
    """Розрізає довгий сигнал на перекривні вікна довжиною window з кроком hop.

    Використовується для прогнозу на довгих власних записах: кожне вікно
    класифікується окремо, а ймовірності усереднюються. Якщо запис коротший
    за одне вікно — повертається одне доповнене нулями вікно.
    """
    if len(signal) <= window:
        return [fix_length(signal, window)]
    starts = range(0, len(signal) - window + 1, hop)
    windows = [signal[s:s + window] for s in starts]
    # Якщо «хвіст» запису не потрапив у жодне вікно — додаємо останнє вікно,
    # вирівняне по кінцю запису, щоб не втратити кінцівку
    if (len(signal) - window) % hop != 0:
        windows.append(signal[-window:])
    return windows
