import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings('ignore')


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


def extract_advanced_meta_features(df):
    """Извлекает расширенные мета-признаки из временных сегментов аудио"""
    print("\nИзвлечение РАСШИРЕННЫХ мета-признаков из аудио сегментов...")

    # Сортируем по файлу и времени для корректного вычисления соседних сегментов
    df = df.sort_values(['file_name', 'start_time']).reset_index(drop=True)
    features_df = df.copy()

    # 1. БАЗОВЫЕ ВРЕМЕННЫЕ ПРИЗНАКИ
    features_df['segment_midpoint'] = (features_df['start_time'] + features_df['end_time']) / 2
    features_df['time_ratio'] = features_df['segment_midpoint'] / features_df['original_length']
    features_df['log_duration'] = np.log1p(features_df['duration'])
    features_df['duration_squared'] = features_df['duration'] ** 2
    features_df['duration_cubed'] = features_df['duration'] ** 3
    features_df['inv_duration'] = 1 / (features_df['duration'] + 0.001)

    # 2. ПРИЗНАКИ СОСЕДНИХ СЕГМЕНТОВ
    features_df['prev_duration'] = features_df.groupby('file_name')['duration'].shift(1)
    features_df['next_duration'] = features_df.groupby('file_name')['duration'].shift(-1)
    features_df['prev_end_time'] = features_df.groupby('file_name')['end_time'].shift(1)

    # Паузы между сегментами
    features_df['silence_before'] = features_df['start_time'] - features_df['prev_end_time']
    features_df['silence_after'] = features_df.groupby('file_name')['start_time'].shift(-1) - features_df['end_time']

    # Заполняем NaN значения
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

    # 9. ВЗАИМОДЕЙСТВИЯ ПРИЗНАКОВ
    features_df['duration_time_interaction'] = features_df['duration'] * features_df['time_ratio']
    features_df['silence_duration_ratio'] = features_df['silence_before'] / (features_df['duration'] + 0.001)
    features_df['complexity_score'] = features_df['file_duration_std'] * features_df['total_segments_in_file']

    # 10. ПРИЗНАКИ НА ОСНОВЕ ГРУППИРОВКИ
    window_size = 3
    features_df['rolling_duration_mean'] = features_df.groupby('file_name')['duration'].rolling(
        window=window_size, min_periods=1).mean().reset_index(drop=True)
    features_df['rolling_duration_std'] = features_df.groupby('file_name')['duration'].rolling(
        window=window_size, min_periods=1).std().reset_index(drop=True)

    # 11. ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ
    features_df['relative_position'] = (features_df['segment_order'] + 1) / features_df['total_segments_in_file']
    features_df['acceleration'] = features_df['duration_change_prev'] - features_df.groupby('file_name')[
        'duration_change_prev'].shift(1)
    features_df['acceleration'] = features_df['acceleration'].fillna(0)
    features_df['has_long_silence_before'] = (features_df['silence_before'] > 0.5).astype(int)
    features_df['has_long_silence_after'] = (features_df['silence_after'] > 0.5).astype(int)
    features_df['is_isolated'] = ((features_df['silence_before'] > 0.3) & (features_df['silence_after'] > 0.3)).astype(
        int)

    print(f"Извлечено {len(features_df.columns) - len(df.columns)} дополнительных мета-признаков")

    # Удаляем временные колонки
    columns_to_drop = ['prev_end_time']
    features_df = features_df.drop(columns=[col for col in columns_to_drop if col in features_df.columns])

    return features_df


def combine_meta_and_audio_features(meta_features_df, audio_features_csv_path):
    """Объединяет мета-признаки и аудио-признаки с диагностикой"""
    print("\nОбъединение мета-признаков и аудио-признаков...")

    # Загружаем аудио-признаки
    audio_features_df = pd.read_csv(audio_features_csv_path)

    # ДИАГНОСТИКА: выводим информацию о данных
    print(f"Мета-признаки: {len(meta_features_df)} сегментов, {meta_features_df['file_name'].nunique()} файлов")
    print(
        f"Аудио-признаки: {len(audio_features_df)} сегментов, {audio_features_df['audio_file_json'].nunique()} файлов")

    # Нормализуем названия файлов для лучшего совпадения
    def normalize_filename(filename):
        """Нормализует названия файлов для лучшего совпадения"""
        # Убираем расширения и лишние символы
        filename = str(filename).lower().replace('.mp3', '').replace('.wav', '')
        # Заменяем разные разделители на одинаковые
        filename = filename.replace('_', ' ').replace('-', ' ')
        # Убираем лишние пробелы
        filename = ' '.join(filename.split())
        return filename

    # Создаем нормализованные версии названий файлов
    meta_features_df['file_name_normalized'] = meta_features_df['file_name'].apply(normalize_filename)
    audio_features_df['audio_file_normalized'] = audio_features_df['audio_file_json'].apply(normalize_filename)

    # Создаем ключи для объединения (с нормализованными именами файлов)
    meta_features_df['merge_key'] = meta_features_df['file_name_normalized'] + '_' + \
                                    meta_features_df['start_time'].round(3).astype(str) + '_' + \
                                    meta_features_df['end_time'].round(3).astype(str)

    audio_features_df['merge_key'] = audio_features_df['audio_file_normalized'] + '_' + \
                                     audio_features_df['start_time'].round(3).astype(str) + '_' + \
                                     audio_features_df['end_time'].round(3).astype(str)

    # Проверяем совпадение ключей
    meta_keys = set(meta_features_df['merge_key'])
    audio_keys = set(audio_features_df['merge_key'])
    common_keys = meta_keys.intersection(audio_keys)

    print(f"Совпадающих ключей: {len(common_keys)}")
    print(f"Процент совпадения: {len(common_keys) / len(meta_keys) * 100:.1f}%")

    # Если совпадений мало, пробуем альтернативные стратегии
    if len(common_keys) < len(meta_keys) * 0.5:  # Меньше 50% совпадений
        print("Мало совпадений, пробуем альтернативные стратегии...")

        # Стратегия 1: объединение только по файлам и приблизительным временам
        combined_alt = []
        for _, meta_row in meta_features_df.iterrows():
            # Ищем соответствующий аудио-сегмент по файлу и близким временам
            audio_match = audio_features_df[
                (audio_features_df['audio_file_normalized'] == meta_row['file_name_normalized']) &
                (abs(audio_features_df['start_time'] - meta_row['start_time']) < 0.1) &
                (abs(audio_features_df['end_time'] - meta_row['end_time']) < 0.1)
                ]

            if len(audio_match) > 0:
                audio_row = audio_match.iloc[0]
                combined_row = {**meta_row.to_dict(), **audio_row.to_dict()}
                combined_alt.append(combined_row)

        if combined_alt:
            combined_df = pd.DataFrame(combined_alt)
            print(f"Альтернативное объединение: {len(combined_df)} сегментов")
            return combined_df

    # Стандартное объединение по ключам
    combined_df = pd.merge(meta_features_df, audio_features_df,
                           on='merge_key', how='inner', suffixes=('_meta', '_audio'))

    print(f"Объединенный датасет: {combined_df.shape[0]} сегментов")
    print(f"Всего признаков: {len(combined_df.columns)}")

    # Анализ потерь по классам
    if 'label' in combined_df.columns:
        original_class_dist = meta_features_df['label'].value_counts()
        combined_class_dist = combined_df['label'].value_counts()

        print(f"\nПОТЕРИ ПО КЛАССАМ ПРИ ОБЪЕДИНЕНИИ:")
        for class_name in original_class_dist.index:
            orig_count = original_class_dist[class_name]
            comb_count = combined_class_dist.get(class_name, 0)
            loss_percent = (1 - comb_count / orig_count) * 100
            print(f"  {class_name}: {orig_count} → {comb_count} ({loss_percent:.1f}% потерь)")

    return combined_df


def clean_and_prepare_data(combined_df):
    """Очистка данных и подготовка признаков"""
    print("\nОчистка данных и подготовка признаков...")

    # Исключаем ненужные колонки
    exclude_columns = ['task_id', 'file_name', 'start_time_meta', 'end_time_meta',
                       'label', 'channel', 'original_length_meta', 'audio_file_json',
                       'audio_file_actual', 'start_time_audio', 'end_time_audio',
                       'labels', 'original_length_audio', 'merge_key']

    feature_columns = [col for col in combined_df.columns if col not in exclude_columns
                       and combined_df[col].dtype in ['int64', 'float64']]

    X = combined_df[feature_columns]
    y = combined_df['label']

    # Анализ пропущенных значений
    print("\nАнализ пропущенных значений:")
    missing_values = X.isnull().sum()
    missing_percent = (missing_values / len(X)) * 100

    missing_info = pd.DataFrame({
        'column': missing_values.index,
        'missing_count': missing_values.values,
        'missing_percent': missing_percent.values
    }).sort_values('missing_percent', ascending=False)

    print(missing_info[missing_info['missing_count'] > 0].head(10))

    # Заполняем пропущенные значения
    print("\nЗаполнение пропущенных значений...")

    # Для числовых признаков
    numeric_columns = X.select_dtypes(include=[np.number]).columns

    # Стратегии заполнения для разных типов признаков
    for col in numeric_columns:
        if X[col].isnull().any():
            # Для признаков с небольшим количеством пропусков используем медиану
            if X[col].isnull().mean() < 0.1:
                X[col] = X[col].fillna(X[col].median())
            else:
                # Для признаков с большим количеством пропусков используем 0
                X[col] = X[col].fillna(0)

    print(f"Пропущенные значения после обработки: {X.isnull().sum().sum()}")

    return X, y, feature_columns


def compare_ensemble_models_with_combined_features(X_train, X_test, y_train, y_test, feature_names, class_names):
    """Сравнение ансамблевых алгоритмов с комбинированными признаками"""

    print("\n" + "=" * 80)
    print(" СРАВНИТЕЛЬНЫЙ АНАЛИЗ АНСАМБЛЕВЫХ АЛГОРИТМОВ (КОМБИНИРОВАННЫЕ ПРИЗНАКИ)")
    print("=" * 80)
    print(f"Многоклассовая классификация: {len(class_names)} классов")
    print(f"Количество признаков: {len(feature_names)}")

    # Базовые модели для ансамблей (с обработкой NaN)
    base_models = {
        'RF': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'SVM': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='linear', random_state=42, probability=True, class_weight='balanced'))
        ]),
        'KNN': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=7))
        ]),
        'LR': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
        ]),
        'HistGB': HistGradientBoostingClassifier(
            random_state=42,
            max_iter=200,
            learning_rate=0.1,
            max_depth=6,
            categorical_features=None
        )
    }

    # Ансамблевые модели
    ensemble_models = {
        # 1. БУСТИНГ (Gradient Boosting) с обработкой NaN
        'Gradient Boosting': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('gb', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ))
        ]),

        # 2. АДАПТИВНЫЙ БУСТИНГ (AdaBoost) с обработкой NaN
        'AdaBoost': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('ab', AdaBoostClassifier(
                n_estimators=200,
                learning_rate=0.1,
                random_state=42
            ))
        ]),

        # 3. EXTRA TREES (случайные поддеревья)
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight='balanced',
            random_state=42
        ),

        # 4. СТЕККИНГ (Stacking) с обработкой NaN
        'Stacking': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('stacking', StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                    ('histgb', HistGradientBoostingClassifier(random_state=42)),
                    ('knn', KNeighborsClassifier(n_neighbors=7))
                ],
                final_estimator=LogisticRegression(random_state=42, class_weight='balanced'),
                cv=3
            ))
        ]),

        # 5. ГОЛОСОВАНИЕ (Voting) - мягкое с обработкой NaN
        'Voting (Soft)': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('voting', VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                    ('histgb', HistGradientBoostingClassifier(random_state=42)),
                    ('knn', KNeighborsClassifier(n_neighbors=7))
                ],
                voting='soft'
            ))
        ]),

        # 6. УЛУЧШЕННЫЙ RANDOM FOREST
        'Enhanced RF': RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),

        # 7. HistGradientBoosting (работает с NaN)
        'HistGradientBoosting': HistGradientBoostingClassifier(
            random_state=42,
            max_iter=200,
            learning_rate=0.1,
            max_depth=8,
            categorical_features=None
        )
    }

    results = []

    # Сначала обучим базовые модели для сравнения
    print("\n БАЗОВЫЕ МОДЕЛИ:")
    for name, model in base_models.items():
        start_time = time.time()
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            training_time = time.time() - start_time

            print(f"  {name}: {accuracy:.4f} ({training_time:.2f} сек)")
        except Exception as e:
            print(f"  {name}: Ошибка - {e}")

    # Теперь ансамблевые модели
    print("\n АНСАМБЛЕВЫЕ МОДЕЛИ:")
    for name, model in ensemble_models.items():
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
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            results.append({
                'Model': name,
                'Type': _get_model_type(name),
                'Accuracy': accuracy,
                'CV Mean': cv_mean,
                'CV Std': cv_std,
                'Training Time': training_time
            })

            print(f"    Точность: {accuracy:.4f}")
            print(f"    Время обучения: {training_time:.2f} сек")
            print(f"    Кросс-валидация: {cv_mean:.4f} ± {cv_std:.4f}")

            # Детальный отчет для моделей с хорошей точностью
            if accuracy > 0.4:
                print(f"    Детальный отчет для {name}:")
                print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

        except Exception as e:
            print(f"    Ошибка в {name}: {e}")
            results.append({
                'Model': name,
                'Type': _get_model_type(name),
                'Accuracy': 0,
                'CV Mean': 0,
                'CV Std': 0,
                'Training Time': 0
            })

    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)

    print("\n" + "=" * 80)
    print(" РЕЙТИНГ АНСАМБЛЕВЫХ АЛГОРИТМОВ (КОМБИНИРОВАННЫЕ ПРИЗНАКИ)")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Визуализация результатов
    _plot_ensemble_results(results_df)

    return results_df, ensemble_models


def _get_model_type(model_name):
    """Определяет тип модели для группировки"""
    if 'Boosting' in model_name or 'Gradient' in model_name:
        return 'Boosting'
    elif 'Voting' in model_name or 'Stacking' in model_name:
        return 'Ensemble'
    elif 'Trees' in model_name or 'RF' in model_name:
        return 'Tree-based'
    else:
        return 'Other'


def _plot_ensemble_results(results_df):
    """Визуализация результатов ансамблевых моделей"""
    plt.figure(figsize=(16, 12))

    # Цвета по типам моделей
    colors = {'Boosting': '#FF6B6B', 'Ensemble': '#4ECDC4', 'Tree-based': '#45B7D1', 'Other': '#96CEB4'}

    # График точности по типам моделей
    plt.subplot(2, 2, 1)
    for model_type in colors.keys():
        type_data = results_df[results_df['Type'] == model_type]
        if len(type_data) > 0:
            plt.barh(type_data['Model'], type_data['Accuracy'],
                     color=colors[model_type], label=model_type, alpha=0.8)

    plt.xlabel('Accuracy')
    plt.title('Точность ансамблевых алгоритмов по типам (Комбинированные признаки)')
    plt.legend()
    plt.xlim(0, 1)

    # График времени обучения
    plt.subplot(2, 2, 2)
    bars = plt.barh(results_df['Model'], results_df['Training Time'],
                    color=[colors[typ] for typ in results_df['Type']])
    plt.xlabel('Training Time (sec)')
    plt.title('Время обучения ансамблевых алгоритмов')

    # График кросс-валидации с ошибками
    plt.subplot(2, 2, 3)
    y_pos = np.arange(len(results_df))
    plt.barh(y_pos, results_df['CV Mean'], xerr=results_df['CV Std'],
             color=[colors[typ] for typ in results_df['Type']], alpha=0.7)
    plt.yticks(y_pos, results_df['Model'])
    plt.xlabel('Cross-Validation Score')
    plt.title('Кросс-валидация ансамблевых алгоритмов (3-fold)')
    plt.xlim(0, 1)

    # Сравнение Accuracy vs CV Score
    plt.subplot(2, 2, 4)
    scatter = plt.scatter(results_df['Accuracy'], results_df['CV Mean'],
                          c=[colors[typ] for typ in results_df['Type']],
                          s=100, alpha=0.7)

    # Добавляем имена моделей к точкам
    for i, row in results_df.iterrows():
        plt.annotate(row['Model'], (row['Accuracy'], row['CV Mean']),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.xlabel('Test Accuracy')
    plt.ylabel('CV Mean Score')
    plt.title('Сравнение Accuracy и Cross-Validation')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # Линия y=x

    # Легенда для типов моделей
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markersize=10, label=model_type)
                       for model_type, color in colors.items()]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.show()


def diagnose_data_issues(meta_features_df, audio_features_csv_path):
    """Диагностика проблем с объединением данных"""
    print("\n" + "=" * 60)
    print("ДИАГНОСТИКА ПРОБЛЕМ С ДАННЫМИ")
    print("=" * 60)

    # Загружаем аудио-признаки
    audio_features_df = pd.read_csv(audio_features_csv_path)

    # Создаем ключи для объединения
    meta_features_df['merge_key'] = meta_features_df['file_name'] + '_' + \
                                    meta_features_df['start_time'].astype(str) + '_' + \
                                    meta_features_df['end_time'].astype(str)

    audio_features_df['merge_key'] = audio_features_df['audio_file_json'] + '_' + \
                                     audio_features_df['start_time'].astype(str) + '_' + \
                                     audio_features_df['end_time'].astype(str)

    print(f"Мета-признаки: {len(meta_features_df)} сегментов")
    print(f"Аудио-признаки: {len(audio_features_df)} сегментов")
    print(f"Уникальных файлов в мета-признаках: {meta_features_df['file_name'].nunique()}")
    print(f"Уникальных файлов в аудио-признаках: {audio_features_df['audio_file_json'].nunique()}")

    # Проверяем совпадение ключей
    meta_keys = set(meta_features_df['merge_key'])
    audio_keys = set(audio_features_df['merge_key'])

    common_keys = meta_keys.intersection(audio_keys)
    unique_meta_keys = meta_keys - audio_keys
    unique_audio_keys = audio_keys - meta_keys

    print(f"\nСОВПАДАЮЩИЕ КЛЮЧИ: {len(common_keys)}")
    print(f"УНИКАЛЬНЫЕ КЛЮЧИ в мета-признаках: {len(unique_meta_keys)}")
    print(f"УНИКАЛЬНЫЕ КЛЮЧИ в аудио-признаках: {len(unique_audio_keys)}")

    # Анализ несовпадающих ключей
    if len(unique_meta_keys) > 0:
        print(f"\nПРИМЕРЫ НЕСОВПАДАЮЩИХ КЛЮЧЕЙ (мета-признаки):")
        for key in list(unique_meta_keys)[:5]:
            print(f"  {key}")

    # Проверяем различия в названиях файлов
    meta_files = set(meta_features_df['file_name'])
    audio_files = set(audio_features_df['audio_file_json'])

    common_files = meta_files.intersection(audio_files)
    unique_meta_files = meta_files - audio_files
    unique_audio_files = audio_files - meta_files

    print(f"\nСОВПАДАЮЩИЕ ФАЙЛЫ: {len(common_files)}")
    print(f"УНИКАЛЬНЫЕ ФАЙЛЫ в мета-признаках: {len(unique_meta_files)}")
    print(f"УНИКАЛЬНЫЕ ФАЙЛЫ в аудио-признаках: {len(unique_audio_files)}")

    if len(unique_meta_files) > 0:
        print(f"\nФАЙЛЫ ТОЛЬКО В МЕТА-ПРИЗНАКАХ:")
        for file in list(unique_meta_files)[:5]:
            print(f"  {file}")

    if len(unique_audio_files) > 0:
        print(f"\nФАЙЛЫ ТОЛЬКО В АУДИО-ПРИЗНАКАХ:")
        for file in list(unique_audio_files)[:5]:
            print(f"  {file}")

    return common_keys


# Основная программа
if __name__ == "__main__":
    # Укажи путь к твоему JSON файлу и CSV с аудио-признаками
    json_file_path = "project-1-at-2025-05-13-11-10-34463d27.json"
    audio_features_csv = "advanced_audio_features.csv"  # Файл из второго скрипта

    print(" КОМБИНИРОВАННАЯ КЛАССИФИКАЦИЯ: МЕТА-ПРИЗНАКИ + АУДИО-ПРИЗНАКИ")
    print("=" * 80)

    # Загрузка данных из JSON
    raw_data = load_data_from_json(json_file_path)

    if raw_data is not None:
        # Извлечение РАСШИРЕННЫХ мета-признаков
        meta_features_data = extract_advanced_meta_features(raw_data)

        # Объединение с аудио-признаками
        if os.path.exists(audio_features_csv):
            combined_data = combine_meta_and_audio_features(meta_features_data, audio_features_csv)

            # Очистка и подготовка данных
            X, y, feature_columns = clean_and_prepare_data(combined_data)

            print(f"\nФИНАЛЬНЫЙ КОМБИНИРОВАННЫЙ ДАТАСЕТ:")
            print(f"Образцов: {X.shape[0]}")
            print(f"Классов: {y.nunique()}")
            print(f"Всего признаков: {len(feature_columns)}")

            # Выводим информацию о всех классах
            class_counts = y.value_counts()
            print(f"\nРАСПРЕДЕЛЕНИЕ КЛАССОВ:")
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count} сегментов")

            # ИСПРАВЛЕНИЕ: Фильтруем классы с недостаточным количеством примеров
            # Оставляем классы с минимум 3 примерами (вместо 5)
            min_samples_per_class = 3
            valid_classes = class_counts[class_counts >= min_samples_per_class].index
            mask = y.isin(valid_classes)
            X_filtered = X[mask]
            y_filtered = y[mask]

            print(f"\nФИЛЬТРАЦИЯ КЛАССОВ:")
            print(f"Минимальное количество примеров на класс: {min_samples_per_class}")
            print(f"Классов до фильтрации: {len(class_counts)}")
            print(f"Классов после фильтрации: {len(valid_classes)}")
            print(f"Сегментов после фильтрации: {len(X_filtered)}")

            # Выводим информацию об оставшихся классах
            filtered_class_counts = y_filtered.value_counts()
            print(f"\nОСТАВШИЕСЯ КЛАССЫ ПОСЛЕ ФИЛЬТРАЦИИ:")
            for class_name, count in filtered_class_counts.items():
                print(f"  {class_name}: {count} сегментов")

            # Кодируем метки
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_filtered)
            class_names = le.classes_

            # Разделение на train/test с проверкой
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_filtered, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
                )
            except ValueError as e:
                print(f"\nПРЕДУПРЕЖДЕНИЕ: Не удалось выполнить стратифицированное разделение: {e}")
                print("Используем обычное разделение без стратификации...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_filtered, y_encoded, test_size=0.3, random_state=42, stratify=None
                )

            print(f"\nДАННЫЕ ДЛЯ ОБУЧЕНИЯ:")
            print(f"Обучающая выборка: {X_train.shape}")
            print(f"Тестовая выборка: {X_test.shape}")
            print(f"Признаков: {len(feature_columns)}")
            print(f"Классов: {len(class_names)}")

            # Проверяем распределение классов в train и test
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            unique_test, counts_test = np.unique(y_test, return_counts=True)

            print(f"\nРАСПРЕДЕЛЕНИЕ КЛАССОВ В TRAIN:")
            for class_idx, count in zip(unique_train, counts_train):
                print(f"  {class_names[class_idx]}: {count} сегментов")

            print(f"\nРАСПРЕДЕЛЕНИЕ КЛАССОВ В TEST:")
            for class_idx, count in zip(unique_test, counts_test):
                print(f"  {class_names[class_idx]}: {count} сегментов")

            # Сравнение ансамблевых моделей с комбинированными признаками
            results_df, ensemble_models = compare_ensemble_models_with_combined_features(
                X_train, X_test, y_train, y_test, feature_columns, class_names
            )

            # Анализ лучшей модели
            if len(results_df) > 0 and results_df.iloc[0]['Accuracy'] > 0:
                best_model_name = results_df.iloc[0]['Model']
                best_model = ensemble_models[best_model_name]

                print(f"\n ЛУЧШАЯ АНСАМБЛЕВАЯ МОДЕЛЬ: {best_model_name}")
                print("=" * 60)

                # Переобучаем лучшую модель
                best_model.fit(X_train, y_train)

                # Финальные предсказания
                y_pred_best = best_model.predict(X_test)
                final_accuracy = accuracy_score(y_test, y_pred_best)

                print(f"Финальная точность: {final_accuracy:.4f}")

                # Матрица ошибок
                plt.figure(figsize=(14, 12))
                cm = confusion_matrix(y_test, y_pred_best)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names, yticklabels=class_names)
                plt.title(f'Матрица ошибок - {best_model_name}\nAccuracy: {final_accuracy:.4f}')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.show()

                print(f"\n ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
                print(f"Лучшая ансамблевая модель: {best_model_name}")
                print(f"Точность: {final_accuracy:.4f}")
                print(f"Количество классов: {len(class_names)}")
                print(f"Количество признаков: {len(feature_columns)}")
                print(f"Типы признаков: Мета-признаки + Аудио-признаки")

            else:
                print(" Не удалось обучить ни одну ансамблевую модель")

        else:
            print(f"Файл с аудио-признаками {audio_features_csv} не найден!")
            print("Сначала запустите скрипт audiofeatures_extraction.py")

    else:
        print(" Не удалось загрузить данные из JSON файла")