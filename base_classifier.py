import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.model_selection import cross_val_score
from collections import Counter


def load_data_from_json(json_file_path):
    """Загружает и преобразует данные из JSON файла в DataFrame"""

    print(f"Загрузка данных из JSON файла: {json_file_path}")

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Создаем список для всех аннотаций
    all_annotations = []

    for task in data:
        task_id = task['id']
        file_name = task['file_upload']

        for annotation in task['annotations']:
            for result in annotation['result']:
                # Извлекаем информацию о сегменте аудио
                if result['type'] == 'labels':
                    segment_info = {
                        'task_id': task_id,
                        'file_name': file_name,
                        'start_time': result['value']['start'],
                        'end_time': result['value']['end'],
                        'duration': result['value']['end'] - result['value']['start'],
                        'label': result['value']['labels'][0],  # берем первый лейбл
                        'channel': result['value']['channel'],
                        'original_length': result['original_length']
                    }
                    all_annotations.append(segment_info)

    # Создаем DataFrame
    df = pd.DataFrame(all_annotations)

    print(f"Загружено {len(df)} сегментов аудио")
    return df


def extract_advanced_audio_features(df):
    """Извлекает расширенные признаки из временных сегментов аудио"""

    print("\nИзвлечение РАСШИРЕННЫХ признаков из аудио сегментов...")

    # Сортируем по файлу и времени для корректного вычисления соседних сегментов
    df = df.sort_values(['file_name', 'start_time']).reset_index(drop=True)

    features_df = df.copy()

    # 1. БАЗОВЫЕ ВРЕМЕННЫЕ ПРИЗНАКИ
    features_df['segment_midpoint'] = (features_df['start_time'] + features_df['end_time']) / 2
    features_df['time_ratio'] = features_df['segment_midpoint'] / features_df['original_length']
    features_df['log_duration'] = np.log1p(features_df['duration'])
    features_df['duration_squared'] = features_df['duration'] ** 2
    features_df['duration_cubed'] = features_df['duration'] ** 3
    features_df['inv_duration'] = 1 / (features_df['duration'] + 0.001)  # избегаем деления на 0

    # 2. ПРИЗНАКИ СОСЕДНИХ СЕГМЕНТОВ
    features_df['prev_duration'] = features_df.groupby('file_name')['duration'].shift(1)
    features_df['next_duration'] = features_df.groupby('file_name')['duration'].shift(-1)
    features_df['prev_end_time'] = features_df.groupby('file_name')['end_time'].shift(1)

    # Паузы между сегментами
    features_df['silence_before'] = features_df['start_time'] - features_df['prev_end_time']
    features_df['silence_after'] = features_df.groupby('file_name')['start_time'].shift(-1) - features_df['end_time']

    # Заполняем NaN значения для первых/последних сегментов
    features_df['silence_before'] = features_df['silence_before'].fillna(0)
    features_df['silence_after'] = features_df['silence_after'].fillna(0)
    features_df['prev_duration'] = features_df['prev_duration'].fillna(features_df['duration'])
    features_df['next_duration'] = features_df['next_duration'].fillna(features_df['duration'])

    # Изменения длительности относительно соседей
    features_df['duration_change_prev'] = features_df['duration'] - features_df['prev_duration']
    features_df['duration_change_next'] = features_df['duration'] - features_df['next_duration']
    features_df['duration_ratio_prev'] = features_df['duration'] / (features_df['prev_duration'] + 0.001)
    features_df['duration_ratio_next'] = features_df['duration'] / (features_df['next_duration'] + 0.001)

    # 3. СТАТИСТИЧЕСКИЕ ПРИЗНАКИ ПО ФАЙЛАМ
    file_stats = df.groupby('file_name').agg({
        'duration': ['mean', 'std', 'min', 'max', 'median'],
        'start_time': ['min', 'max', 'count']
    }).reset_index()

    file_stats.columns = ['file_name', 'file_duration_mean', 'file_duration_std',
                          'file_duration_min', 'file_duration_max', 'file_duration_median',
                          'file_start_min', 'file_start_max', 'total_segments_in_file']

    features_df = features_df.merge(file_stats, on='file_name', how='left')

    # 4. ОТНОСИТЕЛЬНЫЕ ПРИЗНАКИ
    features_df['duration_ratio_to_mean'] = features_df['duration'] / features_df['file_duration_mean']
    features_df['duration_ratio_to_median'] = features_df['duration'] / features_df['file_duration_median']
    features_df['duration_z_score'] = (features_df['duration'] - features_df['file_duration_mean']) / (
                features_df['file_duration_std'] + 0.001)
    features_df['position_in_file'] = (features_df['start_time'] - features_df['file_start_min']) / (
                features_df['file_start_max'] - features_df['file_start_min'] + 0.001)

    # 5. ПРИЗНАКИ НА ОСНОВЕ ПОРЯДКА СЕГМЕНТОВ
    features_df['segment_order'] = features_df.groupby('file_name').cumcount()
    features_df['order_ratio'] = features_df['segment_order'] / features_df['total_segments_in_file']
    features_df['is_first_segment'] = (features_df['segment_order'] == 0).astype(int)
    features_df['is_last_segment'] = (features_df['segment_order'] == features_df['total_segments_in_file'] - 1).astype(
        int)

    # 6. СЕЗОННЫЕ/ПЕРИОДИЧЕСКИЕ ПРИЗНАКИ
    features_df['time_sin'] = np.sin(2 * np.pi * features_df['time_ratio'])
    features_df['time_cos'] = np.cos(2 * np.pi * features_df['time_ratio'])
    features_df['position_sin'] = np.sin(2 * np.pi * features_df['position_in_file'])
    features_df['position_cos'] = np.cos(2 * np.pi * features_df['position_in_file'])

    # 7. КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ ВРЕМЕНИ
    features_df['is_early'] = (features_df['time_ratio'] < 0.33).astype(int)
    features_df['is_middle'] = ((features_df['time_ratio'] >= 0.33) & (features_df['time_ratio'] <= 0.66)).astype(int)
    features_df['is_late'] = (features_df['time_ratio'] > 0.66).astype(int)

    features_df['is_very_short'] = (features_df['duration'] < 0.1).astype(int)
    features_df['is_short'] = ((features_df['duration'] >= 0.1) & (features_df['duration'] < 0.5)).astype(int)
    features_df['is_medium'] = ((features_df['duration'] >= 0.5) & (features_df['duration'] < 1.0)).astype(int)
    features_df['is_long'] = (features_df['duration'] >= 1.0).astype(int)

    # 8. ПРИЗНАКИ РИТМА И ТЕМПА
    features_df['speech_rate_est'] = features_df['total_segments_in_file'] / features_df['file_start_max']
    features_df['avg_segment_duration'] = features_df['file_start_max'] / features_df['total_segments_in_file']
    features_df['tempo_ratio'] = features_df['duration'] / features_df['avg_segment_duration']

    # 9. ВЗАИМОДЕЙСТВИЯ ПРИЗНАКОВ (ИСПРАВЛЕННАЯ ЧАСТЬ)
    features_df['duration_time_interaction'] = features_df['duration'] * features_df['time_ratio']
    features_df['silence_duration_ratio'] = features_df['silence_before'] / (features_df['duration'] + 0.001)

    # Исправляем - используем file_duration_std вместо duration_std
    features_df['complexity_score'] = features_df['file_duration_std'] * features_df['total_segments_in_file']

    # 10. ПРИЗНАКИ НА ОСНОВЕ ГРУППИРОВКИ (скользящие окна)
    window_size = 3
    features_df['rolling_duration_mean'] = features_df.groupby('file_name')['duration'].rolling(
        window=window_size, min_periods=1).mean().reset_index(drop=True)
    features_df['rolling_duration_std'] = features_df.groupby('file_name')['duration'].rolling(
        window=window_size, min_periods=1).std().reset_index(drop=True)

    # 11. ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ
    # Относительная позиция в группе сегментов
    features_df['relative_position'] = (features_df['segment_order'] + 1) / features_df['total_segments_in_file']

    # Признаки на основе скорости изменения
    features_df['acceleration'] = features_df['duration_change_prev'] - features_df.groupby('file_name')[
        'duration_change_prev'].shift(1)
    features_df['acceleration'] = features_df['acceleration'].fillna(0)

    # Бинарные признаки для особых случаев
    features_df['has_long_silence_before'] = (features_df['silence_before'] > 0.5).astype(int)
    features_df['has_long_silence_after'] = (features_df['silence_after'] > 0.5).astype(int)
    features_df['is_isolated'] = ((features_df['silence_before'] > 0.3) & (features_df['silence_after'] > 0.3)).astype(
        int)

    print(f"Извлечено {len(features_df.columns) - len(df.columns)} дополнительных признаков")

    # Удаляем временные колонки, которые не нужны для ML
    columns_to_drop = ['prev_end_time']
    features_df = features_df.drop(columns=[col for col in columns_to_drop if col in features_df.columns])

    # Заполняем оставшиеся NaN значения
    features_df = features_df.fillna(0)

    return features_df


def analyze_class_distribution(df, title="Распределение классов"):
    """Анализ и визуализация распределения классов"""
    class_counts = Counter(df['label'])

    print(f"\n{title}:")
    print("=" * 50)

    classes_sorted = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for class_name, count in classes_sorted:
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

    # Визуализация
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    classes, counts = zip(*classes_sorted)
    bars = plt.bar(range(len(classes)), counts, color='skyblue')
    plt.xlabel('Классы')
    plt.ylabel('Количество образцов')
    plt.title('Распределение классов')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')

    # Добавляем значения на столбцы
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(count), ha='center', va='bottom', fontsize=8)

    plt.subplot(1, 2, 2)
    # Логарифмическая шкала для лучшей визуализации
    plt.bar(range(len(classes)), counts, color='lightcoral')
    plt.yscale('log')
    plt.xlabel('Классы')
    plt.ylabel('Количество (log scale)')
    plt.title('Распределение классов (логарифмическая шкала)')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    return class_counts


def filter_rare_classes(df, min_samples_per_class=5):
    """Фильтрация редких классов"""
    class_counts = df['label'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index

    filtered_data = df[df['label'].isin(valid_classes)]

    print(f"\nФИЛЬТРАЦИЯ РЕДКИХ КЛАССОВ:")
    print(f"Минимальное количество образцов на класс: {min_samples_per_class}")
    print(f"Исходно: {df.shape[0]} образцов, {len(class_counts)} классов")
    print(f"После фильтрации: {filtered_data.shape[0]} образцов, {len(valid_classes)} классов")
    print(f"Удалено классов: {len(class_counts) - len(valid_classes)}")

    return filtered_data


def prepare_features_for_ml(df):
    """Подготовка признаков для машинного обучения"""

    # Исключаем не-признаковые колонки
    exclude_columns = ['task_id', 'file_name', 'start_time', 'end_time', 'label', 'channel', 'original_length']

    # Все числовые колонки кроме исключенных
    feature_columns = [col for col in df.columns if
                       col not in exclude_columns and df[col].dtype in ['int64', 'float64']]

    print(f"Используемые признаки для ML: {len(feature_columns)} признаков")
    print("Первые 20 признаков:", feature_columns[:20])

    X = df[feature_columns]
    y = df['label']

    return X, y, feature_columns


def compare_ml_models(X_train, X_test, y_train, y_test, feature_names, class_names):
    """Сравнение различных алгоритмов машинного обучения для многоклассовой классификации"""

    print("\n" + "=" * 70)
    print(" СРАВНИТЕЛЬНЫЙ АНАЛИЗ АЛГОРИТМОВ МАШИННОГО ОБУЧЕНИЯ")
    print("=" * 70)
    print(f"Многоклассовая классификация: {len(class_names)} классов")
    print(f"Количество признаков: {len(feature_names)}")

    # Определяем модели для сравнения (настроены для многоклассовой классификации)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced',
                                                max_depth=20),
        'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True, class_weight='balanced'),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'Gaussian Naive Bayes': GaussianNB()
    }

    results = []

    for name, model in models.items():
        print(f"\n Обучение {name}...")
        start_time = time.time()

        try:
            # Обучение модели
            model.fit(X_train, y_train)

            # Предсказания
            y_pred = model.predict(X_test)

            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            training_time = time.time() - start_time

            # Кросс-валидация
            cv_scores = cross_val_score(model, X_train, y_train, cv=3,
                                        scoring='accuracy')  # Уменьшили cv из-за редких классов
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'CV Mean': cv_mean,
                'CV Std': cv_std,
                'Training Time': training_time
            })

            print(f"    Точность: {accuracy:.4f}")
            print(f"    Время обучения: {training_time:.2f} сек")
            print(f"    Кросс-валидация: {cv_mean:.4f} ± {cv_std:.4f}")

            # Детальный отчет для моделей с хорошей точностью
            if accuracy > 0.3:
                print(f"   📈 Детальный отчет для {name}:")
                print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

        except Exception as e:
            print(f"    Ошибка в {name}: {e}")
            results.append({
                'Model': name,
                'Accuracy': 0,
                'CV Mean': 0,
                'CV Std': 0,
                'Training Time': 0
            })

    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)

    print("\n" + "=" * 70)
    print(" РЕЙТИНГ АЛГОРИТМОВ ПО ТОЧНОСТИ")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Визуализация результатов
    plt.figure(figsize=(15, 10))

    # График точности
    plt.subplot(2, 2, 1)
    bars = plt.barh(results_df['Model'], results_df['Accuracy'], color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Сравнение точности алгоритмов\n(многоклассовая классификация)')
    plt.xlim(0, 1)

    # Добавляем значения на bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{width:.3f}', ha='left', va='center')

    # График времени обучения
    plt.subplot(2, 2, 2)
    plt.barh(results_df['Model'], results_df['Training Time'], color='lightcoral')
    plt.xlabel('Training Time (sec)')
    plt.title('Время обучения алгоритмов')

    # График кросс-валидации
    plt.subplot(2, 2, 3)
    plt.barh(results_df['Model'], results_df['CV Mean'],
             xerr=results_df['CV Std'], color='lightgreen', alpha=0.7)
    plt.xlabel('Cross-Validation Score')
    plt.title('Кросс-валидация (3-fold)')
    plt.xlim(0, 1)

    plt.tight_layout()
    plt.show()

    return results_df, models


def plot_feature_importance(best_model, feature_names, class_names, top_n=20):
    """Визуализация важности признаков"""
    plt.figure(figsize=(12, 8))

    if hasattr(best_model, 'feature_importances_'):
        # Random Forest
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=True).tail(top_n)

        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title(f'Топ-{top_n} самых важных признаков (Random Forest)')
        plt.xlabel('Важность')

    elif hasattr(best_model, 'coef_'):
        # Линейные модели
        if len(best_model.coef_.shape) > 1:
            # Для многоклассовой - берем среднее по классам
            coef_mean = np.mean(np.abs(best_model.coef_), axis=0)
        else:
            coef_mean = np.abs(best_model.coef_)

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': coef_mean
        }).sort_values('importance', ascending=True).tail(top_n)

        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title(f'Топ-{top_n} самых важных признаков (коэффициенты)')
        plt.xlabel('Абсолютное значение коэффициента')

    else:
        plt.text(0.5, 0.5, 'Важность признаков недоступна\nдля этого алгоритма',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Важность признаков')

    plt.tight_layout()
    plt.show()


# Основная программа
if __name__ == "__main__":
    # Укажи путь к твоему JSON файлу
    json_file_path = "project-1-at-2025-05-13-11-10-34463d27.json"  # замени на актуальный путь

    print(" МНОГОКЛАССОВАЯ КЛАССИФИКАЦИЯ АУДИО СЕГМЕНТОВ С РАСШИРЕННЫМИ ПРИЗНАКАМИ")
    print("=" * 80)

    # Загрузка данных из JSON
    raw_data = load_data_from_json(json_file_path)

    if raw_data is not None:
        # Анализ исходного распределения классов
        analyze_class_distribution(raw_data, "Исходное распределение классов")

        # Извлечение РАСШИРЕННЫХ признаков
        features_data = extract_advanced_audio_features(raw_data)

        # Фильтрация редких классов
        filtered_data = filter_rare_classes(features_data, min_samples_per_class=5)

        # Анализ после фильтрации
        analyze_class_distribution(filtered_data, "Распределение классов после фильтрации")

        # Подготовка данных для ML
        X, y, feature_names = prepare_features_for_ml(filtered_data)

        print(f"\nФИНАЛЬНЫЙ ДАТАСЕТ ДЛЯ ML:")
        print(f"Размер: {X.shape}")
        print(f"Количество признаков: {len(feature_names)}")
        print(f"Количество классов: {len(set(y))}")

        # Кодируем метки
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_

        print(f"\nКОДИРОВАНИЕ КЛАССОВ:")
        for i, class_name in enumerate(class_names):
            count = sum(y == class_name)
            print(f"  {class_name} -> {i} ({count} samples)")

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )

        print(f"\nРАЗДЕЛЕНИЕ НА TRAIN/TEST:")
        print(f"Обучающая: {X_train.shape[0]} samples")
        print(f"Тестовая: {X_test.shape[0]} samples")

        # Масштабирование признаков
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\nМАСШТАБИРОВАНИЕ ПРИЗНАКОВ:")
        print(f"Размерность после масштабирования: {X_train_scaled.shape}")

        # Сравнение моделей
        results_df, models = compare_ml_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_names,
                                               class_names)

        # Анализ лучшей модели
        if len(results_df) > 0 and results_df.iloc[0]['Accuracy'] > 0:
            best_model_name = results_df.iloc[0]['Model']
            best_model = models[best_model_name]

            print(f"\n ЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
            print("=" * 50)

            # Переобучаем лучшую модель на всех данных
            best_model.fit(X_train_scaled, y_train)

            # Предсказания и оценка
            y_pred_best = best_model.predict(X_test_scaled)
            final_accuracy = accuracy_score(y_test, y_pred_best)

            print(f"Финальная точность: {final_accuracy:.4f}")

            # Визуализация важности признаков
            plot_feature_importance(best_model, feature_names, class_names, top_n=20)

            print(f"\nФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
            print(f"Лучшая модель: {best_model_name}")
            print(f"Точность: {final_accuracy:.4f}")
            print(f"Количество классов: {len(class_names)}")
            print(f"Количество признаков: {len(feature_names)}")

            # Сохраняем обработанные данные
            filtered_data.to_csv('processed_audio_segments_advanced.csv', index=False)
            print(f"\n Обработанные данные сохранены: processed_audio_segments_advanced.csv")

        else:
            print(" Не удалось обучить ни одну модель")

    else:
        print(" Не удалось загрузить данные из JSON файла")