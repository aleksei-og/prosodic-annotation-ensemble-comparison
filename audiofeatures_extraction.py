import json
import os
import pandas as pd
import numpy as np
import librosa
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import hilbert
import pywt
from tqdm import tqdm
import re

# Пути к файлам
JSON_PATH = "project-1-at-2025-05-13-11-10-34463d27.json"
AUDIO_DIR = r"C:\Users\aleks\Desktop\jupyter without datasets\audiosets"
OUTPUT_CSV = "advanced_audio_features.csv"


def find_audio_file(audio_dir, filename_from_json):
    """Находит аудиофайл по частичному совпадению названия"""
    # Извлекаем основное название из JSON (часть после дефиса)
    match = re.search(r'-(.+)\.mp3$', filename_from_json)
    if match:
        search_pattern = match.group(1)  # Например: "In_a_restaurant"
    else:
        search_pattern = filename_from_json.replace('.mp3', '')

    # Ищем файлы в директории
    for file in os.listdir(audio_dir):
        if file.endswith('.mp3') and search_pattern in file:
            return os.path.join(audio_dir, file)

    # Если не нашли по частичному совпадению, попробуем найти любой файл с похожим названием
    simple_name = search_pattern.replace('_', ' ').lower()
    for file in os.listdir(audio_dir):
        if file.endswith('.mp3'):
            file_simple = file.replace('.mp3', '').replace('_', ' ').lower()
            if simple_name in file_simple or file_simple in simple_name:
                return os.path.join(audio_dir, file)

    return None


def extract_advanced_audio_features(audio_path, start_time, end_time, sr=22050):
    """Расширенное извлечение аудио-признаков из сегмента"""
    try:
        # Загрузка аудио файла
        y, sr = librosa.load(audio_path, sr=sr)

        # Вычисление временных меток в сэмплах
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Извлечение сегмента
        segment = y[start_sample:end_sample]

        if len(segment) < 512:  # Минимальная длина для анализа
            return None

        features = {}

        # ===== БАЗОВЫЕ СТАТИСТИКИ СИГНАЛА =====
        features['amplitude_mean'] = np.mean(segment)
        features['amplitude_std'] = np.std(segment)
        features['amplitude_skew'] = skew(segment)
        features['amplitude_kurtosis'] = kurtosis(segment)
        features['amplitude_max'] = np.max(segment)
        features['amplitude_min'] = np.min(segment)
        features['amplitude_range'] = np.ptp(segment)

        # ===== ЭНЕРГЕТИЧЕСКИЕ ПРИЗНАКИ =====
        features['rms'] = np.mean(librosa.feature.rms(y=segment))
        features['energy'] = np.sum(segment ** 2)

        # ===== ВРЕМЕННЫЕ ПРИЗНАКИ =====
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(segment))
        features['zcr_std'] = np.std(librosa.feature.zero_crossing_rate(segment))

        # ===== СПЕКТРАЛЬНЫЕ ПРИЗНАКИ =====
        stft = np.abs(librosa.stft(segment))

        # Спектральные центроиды
        spectral_centroids = librosa.feature.spectral_centroid(y=segment, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)

        # Спектральная ширина полосы
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)

        # Спектральный спад
        spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr, roll_percent=0.85)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)

        # Спектральная плоскость
        spectral_flatness = librosa.feature.spectral_flatness(y=segment)
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)

        # ===== MFCC ПРИЗНАКИ (13 коэффициентов) =====
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i + 1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i + 1}_std'] = np.std(mfccs[i])

        # ===== ХРОМАТИЧЕСКИЕ ПРИЗНАКИ =====
        chroma_stft = librosa.feature.chroma_stft(y=segment, sr=sr)
        features['chroma_stft_mean'] = np.mean(chroma_stft)
        features['chroma_stft_std'] = np.std(chroma_stft)

        # ===== МЕЛ-СПЕКТРОГРАММА =====
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
        features['mel_spectrogram_mean'] = np.mean(mel_spec)
        features['mel_spectrogram_std'] = np.std(mel_spec)

        # ===== ПРИЗНАКИ ГАРМОНИК И ПЕРКУССИИ =====
        y_harmonic, y_percussive = librosa.effects.hpss(segment)
        features['harmonic_ratio'] = np.mean(y_harmonic ** 2) / (
                    np.mean(y_harmonic ** 2) + np.mean(y_percussive ** 2) + 1e-10)

        # ===== ТЕМПО И РИТМИЧЕСКИЕ ПРИЗНАКИ =====
        tempo, beats = librosa.beat.beat_track(y=segment, sr=sr)
        features['tempo'] = tempo

        # ===== ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ =====
        features['duration'] = end_time - start_time
        features['sample_length'] = len(segment)
        features['non_silence_ratio'] = np.mean(np.abs(segment) > 0.01)

        return features

    except Exception as e:
        print(f"Ошибка при обработке {os.path.basename(audio_path)} [{start_time:.2f}-{end_time:.2f}s]: {str(e)}")
        return None


def process_json_data(json_path, audio_dir):
    """Обработка JSON файла и извлечение признаков"""

    # Загрузка JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_features = []
    found_files = {}

    print("Поиск аудиофайлов...")
    for item in tqdm(data, desc="Поиск файлов"):
        audio_filename_json = item['file_upload']
        audio_path = find_audio_file(audio_dir, audio_filename_json)

        if audio_path:
            found_files[audio_filename_json] = os.path.basename(audio_path)
        else:
            print(f"Файл не найден для: {audio_filename_json}")
            continue

    print(f"\nНайдено файлов: {len(found_files)} из {len(data)}")

    # Обработка найденных файлов
    for item in tqdm(data, desc="Извлечение признаков"):
        audio_filename_json = item['file_upload']

        if audio_filename_json not in found_files:
            continue

        audio_path = os.path.join(audio_dir, found_files[audio_filename_json])

        # Обработка аннотаций
        for annotation in item['annotations']:
            for result in annotation['result']:
                if result['type'] == 'labels':
                    # Извлечение временных меток и меток
                    start_time = result['value']['start']
                    end_time = result['value']['end']
                    labels = result['value']['labels']

                    # Извлечение признаков
                    features = extract_advanced_audio_features(audio_path, start_time, end_time)

                    if features is not None:
                        # Добавление метаданных
                        features['audio_file_json'] = audio_filename_json
                        features['audio_file_actual'] = found_files[audio_filename_json]
                        features['start_time'] = start_time
                        features['end_time'] = end_time
                        features['labels'] = '|'.join(labels)
                        features['task_id'] = item['id']
                        features['original_length'] = result.get('original_length', 0)

                        all_features.append(features)

    return all_features


# Основной процесс
print("Начало извлечения расширенных аудио-признаков...")
print("Директория с аудио:", AUDIO_DIR)

# Проверим какие файлы есть в директории
print("\nФайлы в аудио директории:")
audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')]
for file in audio_files:
    print(f"  - {file}")

print(f"\nВсего найдено {len(audio_files)} MP3 файлов")

# Извлечение признаков
features_list = process_json_data(JSON_PATH, AUDIO_DIR)

if features_list:
    # Создание DataFrame
    df = pd.DataFrame(features_list)

    # Сохранение в CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    print(f"\n✅ Готово! Извлечено {len(df)} сегментов")
    print(f"📁 Файл сохранен как: {OUTPUT_CSV}")
    print(f"📊 Размерность данных: {df.shape}")

    # Статистика по меткам
    print(f"\n🏷️ Распределение меток:")
    label_counts = df['labels'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count} сегментов")

    print(f"\n📈 Количество признаков: {len(df.columns)}")
    print("\nПервые 3 сегмента:")
    print(df[['audio_file_actual', 'start_time', 'end_time', 'labels', 'duration']].head(3))

else:
    print("❌ Не удалось извлечь признаки.")

print(f"\nОбщее количество извлеченных признаков: {len(df.columns) if features_list else 0}")