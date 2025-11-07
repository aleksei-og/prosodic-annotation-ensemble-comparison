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
    ExtraTreesClassifier
)
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

    print(f"Извлечено {len(features_df.columns) - len(df.columns)} дополнительных признаков")

    # Удаляем временные колонки
    columns_to_drop = ['prev_end_time']
    features_df = features_df.drop(columns=[col for col in columns_to_drop if col in features_df.columns])

    # Заполняем оставшиеся NaN значения
    features_df = features_df.fillna(0)

    return features_df


def compare_ensemble_models(X_train, X_test, y_train, y_test, feature_names, class_names):
    """Сравнение ансамблевых алгоритмов машинного обучения"""

    print("\n" + "=" * 80)
    print(" СРАВНИТЕЛЬНЫЙ АНАЛИЗ АНСАМБЛЕВЫХ АЛГОРИТМОВ")
    print("=" * 80)
    print(f"Многоклассовая классификация: {len(class_names)} классов")
    print(f"Количество признаков: {len(feature_names)}")

    # Базовые модели для ансамблей
    base_models = {
        'RF': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'SVM': SVC(kernel='linear', random_state=42, probability=True, class_weight='balanced'),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'LR': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    }

    # Ансамблевые модели
    ensemble_models = {
        # 1. БУСТИНГ (Gradient Boosting)
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),

        # 2. АДАПТИВНЫЙ БУСТИНГ (AdaBoost)
        'AdaBoost': AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.1,
            random_state=42
        ),

        # 3. EXTRA TREES (случайные поддеревья)
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight='balanced',
            random_state=42
        ),

        # 4. СТЕККИНГ (Stacking)
        'Stacking': StackingClassifier(
            estimators=[
                ('rf', base_models['RF']),
                ('svm', base_models['SVM']),
                ('knn', base_models['KNN'])
            ],
            final_estimator=LogisticRegression(random_state=42, class_weight='balanced'),
            cv=3
        ),

        # 5. ГОЛОСОВАНИЕ (Voting) - мягкое
        'Voting (Soft)': VotingClassifier(
            estimators=[
                ('rf', base_models['RF']),
                ('svm', base_models['SVM']),
                ('knn', base_models['KNN'])
            ],
            voting='soft',
            weights=[2, 1, 1]  # Больший вес для Random Forest
        ),

        # 6. УЛУЧШЕННЫЙ RANDOM FOREST
        'Enhanced RF': RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
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
    print(" РЕЙТИНГ АНСАМБЛЕВЫХ АЛГОРИТМОВ")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Визуализация результатов
    _plot_ensemble_results(results_df)

    return results_df, ensemble_models


def _get_model_type(model_name):
    """Определяет тип модели для группировки"""
    if 'Boosting' in model_name:
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
    plt.title('Точность ансамблевых алгоритмов по типам')
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


def analyze_feature_importance_ensemble(best_ensemble_model, feature_names, top_n=25):
    """Анализ важности признаков для ансамблевых моделей"""

    plt.figure(figsize=(12, 10))

    if hasattr(best_ensemble_model, 'feature_importances_'):
        # Для моделей с feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_ensemble_model.feature_importances_
        }).sort_values('importance', ascending=True).tail(top_n)

        plt.barh(importance_df['feature'], importance_df['importance'], color='lightcoral')
        plt.title(f'Топ-{top_n} самых важных признаков\n({type(best_ensemble_model).__name__})')
        plt.xlabel('Важность признака')

    elif hasattr(best_ensemble_model, 'estimators_'):
        # Для ансамблевых моделей типа Random Forest
        importances = []
        for estimator in best_ensemble_model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                importances.append(estimator.feature_importances_)

        if importances:
            mean_importance = np.mean(importances, axis=0)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_importance
            }).sort_values('importance', ascending=True).tail(top_n)

            plt.barh(importance_df['feature'], importance_df['importance'], color='lightgreen')
            plt.title(f'Топ-{top_n} самых важных признаков\n(Среднее по ансамблю)')
            plt.xlabel('Средняя важность признака')

    elif hasattr(best_ensemble_model, 'coef_'):
        # Для линейных моделей в ансамблях
        if len(best_ensemble_model.coef_.shape) > 1:
            coef_mean = np.mean(np.abs(best_ensemble_model.coef_), axis=0)
        else:
            coef_mean = np.abs(best_ensemble_model.coef_)

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': coef_mean
        }).sort_values('importance', ascending=True).tail(top_n)

        plt.barh(importance_df['feature'], importance_df['importance'], color='lightblue')
        plt.title(f'Топ-{top_n} самых важных признаков\n(Абсолютные коэффициенты)')
        plt.xlabel('Абсолютное значение коэффициента')

    else:
        plt.text(0.5, 0.5, 'Важность признаков недоступна\nдля этого типа ансамбля',
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Анализ важности признаков')

    plt.tight_layout()
    plt.show()


# Основная программа
if __name__ == "__main__":
    # Укажи путь к твоему JSON файлу
    json_file_path = "project-1-at-2025-05-13-11-10-34463d27.json"

    print(" АНСАМБЛЕВАЯ КЛАССИФИКАЦИЯ АУДИО СЕГМЕНТОВ")
    print("=" * 80)

    # Загрузка данных из JSON
    raw_data = load_data_from_json(json_file_path)

    if raw_data is not None:
        # Извлечение РАСШИРЕННЫХ признаков
        features_data = extract_advanced_audio_features(raw_data)

        # Фильтрация редких классов
        class_counts = features_data['label'].value_counts()
        valid_classes = class_counts[class_counts >= 5].index
        filtered_data = features_data[features_data['label'].isin(valid_classes)]

        print(f"\nФИНАЛЬНЫЙ ДАТАСЕТ:")
        print(f"Образцов: {filtered_data.shape[0]}")
        print(f"Классов: {len(valid_classes)}")

        # Подготовка данных для ML
        exclude_columns = ['task_id', 'file_name', 'start_time', 'end_time', 'label', 'channel', 'original_length']
        feature_columns = [col for col in filtered_data.columns if col not in exclude_columns
                           and filtered_data[col].dtype in ['int64', 'float64']]

        X = filtered_data[feature_columns]
        y = filtered_data['label']

        # Кодируем метки
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )

        # Масштабирование признаков
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\nДАННЫЕ ДЛЯ ОБУЧЕНИЯ:")
        print(f"Обучающая выборка: {X_train_scaled.shape}")
        print(f"Тестовая выборка: {X_test_scaled.shape}")
        print(f"Признаков: {len(feature_columns)}")
        print(f"Классов: {len(class_names)}")

        # Сравнение ансамблевых моделей
        results_df, ensemble_models = compare_ensemble_models(
            X_train_scaled, X_test_scaled, y_train, y_test, feature_columns, class_names
        )

        # Анализ лучшей модели
        if len(results_df) > 0 and results_df.iloc[0]['Accuracy'] > 0:
            best_model_name = results_df.iloc[0]['Model']
            best_model = ensemble_models[best_model_name]

            print(f"\n ЛУЧШАЯ АНСАМБЛЕВАЯ МОДЕЛЬ: {best_model_name}")
            print("=" * 60)

            # Переобучаем лучшую модель
            best_model.fit(X_train_scaled, y_train)

            # Финальные предсказания
            y_pred_best = best_model.predict(X_test_scaled)
            final_accuracy = accuracy_score(y_test, y_pred_best)

            print(f"Финальная точность: {final_accuracy:.4f}")

            # Анализ важности признаков
            print(f"\n АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ ДЛЯ {best_model_name}:")
            analyze_feature_importance_ensemble(best_model, feature_columns, top_n=20)

            # Матрица ошибок
            plt.figure(figsize=(12, 10))
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
            print(f"Улучшение относительно базового RF: +{(final_accuracy - 0.4157) * 100:.2f}%")
            print(f"Количество классов: {len(class_names)}")
            print(f"Количество признаков: {len(feature_columns)}")

        else:
            print(" Не удалось обучить ни одну ансамблевую модель")

    else:
        print(" Не удалось загрузить данные из JSON файла")