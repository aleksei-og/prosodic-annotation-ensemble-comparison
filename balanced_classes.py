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
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.utils.class_weight import compute_class_weight

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


def apply_class_balancing(X_train, y_train, method='smote'):
    """Применяет различные методы балансировки классов"""

    print(f"\n БАЛАНСИРОВКА КЛАССОВ: {method.upper()}")
    print("Распределение классов до балансировки:")
    class_counts_before = dict(zip(*np.unique(y_train, return_counts=True)))
    print(class_counts_before)

    if method == 'smote':
        balancer = SMOTE(random_state=42, k_neighbors=3)
    elif method == 'adasyn':
        balancer = ADASYN(random_state=42, n_neighbors=3)
    elif method == 'smoteenn':
        balancer = SMOTEENN(random_state=42)
    elif method == 'undersample':
        balancer = RandomUnderSampler(random_state=42)
    else:
        return X_train, y_train

    X_balanced, y_balanced = balancer.fit_resample(X_train, y_train)

    print("Распределение классов после балансировки:")
    class_counts_after = dict(zip(*np.unique(y_balanced, return_counts=True)))
    print(class_counts_after)
    print(f"Увеличение выборки: {len(X_balanced)} samples (+{len(X_balanced) - len(X_train)})")

    return X_balanced, y_balanced


def compute_balanced_class_weights(y):
    """Вычисляет сбалансированные веса классов"""
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return dict(zip(np.unique(y), class_weights))


def compare_ensemble_models_balanced(X_train, X_test, y_train, y_test, feature_names, class_names):
    """Сравнение ансамблевых алгоритмов с балансировкой классов"""

    print("\n" + "=" * 80)
    print(" СРАВНИТЕЛЬНЫЙ АНАЛИЗ АНСАМБЛЕВЫХ АЛГОРИТМОВ С БАЛАНСИРОВКОЙ")
    print("=" * 80)
    print(f"Многоклассовая классификация: {len(class_names)} классов")
    print(f"Количество признаков: {len(feature_names)}")

    # Вычисляем веса классов
    class_weights = compute_balanced_class_weights(y_train)
    print("Веса классов:", class_weights)

    # Базовые модели с балансировкой
    base_models = {
        'RF': RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights),
        'SVM': SVC(kernel='linear', random_state=42, probability=True, class_weight=class_weights),
        'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
        'LR': LogisticRegression(random_state=42, max_iter=1000, class_weight=class_weights)
    }

    # Ансамблевые модели с улучшенной балансировкой
    ensemble_models = {


        'AdaBoost': AdaBoostClassifier(
            n_estimators=200, learning_rate=0.1, random_state=42
        ),

        'Balanced RF': RandomForestClassifier(
            n_estimators=300, max_depth=25, min_samples_split=5,
            min_samples_leaf=2, class_weight=class_weights, random_state=42
        ),

        'Balanced Extra Trees': ExtraTreesClassifier(
            n_estimators=200, max_depth=20, class_weight=class_weights, random_state=42
        ),

        'Weighted Voting': VotingClassifier(
            estimators=[
                ('rf', base_models['RF']),
                ('svm', base_models['SVM']),
                ('knn', base_models['KNN'])
            ],
            voting='soft',
            weights=[3, 1, 2]
        ),

        'Stacking Balanced': StackingClassifier(
            estimators=[
                ('rf', base_models['RF']),
                ('svm', base_models['SVM']),
                ('knn', base_models['KNN'])
            ],
            final_estimator=LogisticRegression(random_state=42, class_weight=class_weights),
            cv=3
        )
    }

    # Тестируем разные методы балансировки
    balancing_methods = ['smote', 'adasyn', 'none']

    all_results = []

    for balance_method in balancing_methods:
        print(f"\n🔧 МЕТОД БАЛАНСИРОВКИ: {balance_method.upper()}")

        if balance_method == 'none':
            X_bal, y_bal = X_train, y_train
        else:
            X_bal, y_bal = apply_class_balancing(X_train, y_train, balance_method)

        for name, model in ensemble_models.items():
            print(f"   Обучение {name}...")
            start_time = time.time()

            try:
                model.fit(X_bal, y_bal)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                training_time = time.time() - start_time

                # Кросс-валидация
                cv_scores = cross_val_score(model, X_bal, y_bal, cv=3, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                all_results.append({
                    'Model': name,
                    'Balancing': balance_method,
                    'Accuracy': accuracy,
                    'CV Mean': cv_mean,
                    'CV Std': cv_std,
                    'Training Time': training_time
                })

                print(f"     Точность: {accuracy:.4f}")

                # Детальный отчет для моделей с хорошей точностью
                if accuracy > 0.4:
                    print(f"     Детальный отчет для {name}:")
                    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

            except Exception as e:
                print(f"     Ошибка в {name}: {e}")
                all_results.append({
                    'Model': name,
                    'Balancing': balance_method,
                    'Accuracy': 0,
                    'CV Mean': 0,
                    'CV Std': 0,
                    'Training Time': 0
                })

    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('Accuracy', ascending=False)

    print("\n" + "=" * 80)
    print(" РЕЙТИНГ МОДЕЛЕЙ С БАЛАНСИРОВКОЙ")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Визуализация результатов
    _plot_balanced_results(results_df)

    return results_df, ensemble_models


def _plot_balanced_results(results_df):
    """Визуализация результатов с балансировкой"""

    plt.figure(figsize=(18, 10))

    # Цвета по методам балансировки
    colors = {'smote': '#FF6B6B', 'adasyn': '#4ECDC4', 'none': '#45B7D1'}

    # График точности по методам балансировки
    plt.subplot(2, 2, 1)
    for balance_method in colors.keys():
        method_data = results_df[results_df['Balancing'] == balance_method]
        if len(method_data) > 0:
            plt.barh(method_data['Model'], method_data['Accuracy'],
                     color=colors[balance_method], label=balance_method, alpha=0.8)

    plt.xlabel('Accuracy')
    plt.title('Точность моделей по методам балансировки')
    plt.legend()
    plt.xlim(0, 1)

    # График сравнения методов балансировки
    plt.subplot(2, 2, 2)
    balance_means = results_df.groupby('Balancing')['Accuracy'].mean()
    balance_stds = results_df.groupby('Balancing')['Accuracy'].std()

    bars = plt.bar(balance_means.index, balance_means.values,
                   yerr=balance_stds.values, capsize=5,
                   color=[colors[method] for method in balance_means.index])
    plt.ylabel('Средняя точность')
    plt.title('Сравнение методов балансировки')

    # Добавляем значения на столбцы
    for bar, value in zip(bars, balance_means.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    # Лучшие модели для каждого метода балансировки
    plt.subplot(2, 2, 3)
    best_models = results_df.loc[results_df.groupby('Balancing')['Accuracy'].idxmax()]

    plt.barh(best_models['Model'] + " (" + best_models['Balancing'] + ")",
             best_models['Accuracy'],
             color=[colors[method] for method in best_models['Balancing']])
    plt.xlabel('Accuracy')
    plt.title('Лучшие модели для каждого метода балансировки')
    plt.xlim(0, 1)

    # Время обучения
    plt.subplot(2, 2, 4)
    for balance_method in colors.keys():
        method_data = results_df[results_df['Balancing'] == balance_method]
        if len(method_data) > 0:
            plt.barh(method_data['Model'], method_data['Training Time'],
                     color=colors[balance_method], label=balance_method, alpha=0.6)

    plt.xlabel('Training Time (sec)')
    plt.title('Время обучения по методам балансировки')
    plt.legend()

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

    print(" АНСАМБЛЕВАЯ КЛАССИФИКАЦИЯ АУДИО СЕГМЕНТОВ С БАЛАНСИРОВКОЙ КЛАССОВ")
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

        # Сравнение ансамблевых моделей с балансировкой
        results_df, ensemble_models = compare_ensemble_models_balanced(
            X_train_scaled, X_test_scaled, y_train, y_test, feature_columns, class_names
        )

        # Анализ лучшей модели
        if len(results_df) > 0 and results_df.iloc[0]['Accuracy'] > 0:
            best_result = results_df.iloc[0]
            best_model_name = best_result['Model']
            best_balancing_method = best_result['Balancing']
            best_model = ensemble_models[best_model_name]

            print(f"\n ЛУЧШАЯ МОДЕЛЬ: {best_model_name} с балансировкой {best_balancing_method}")
            print("=" * 60)

            # Переобучаем лучшую модель с лучшим методом балансировки
            if best_balancing_method == 'none':
                X_bal, y_bal = X_train_scaled, y_train
            else:
                X_bal, y_bal = apply_class_balancing(X_train_scaled, y_train, best_balancing_method)

            best_model.fit(X_bal, y_bal)

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
            plt.title(f'Матрица ошибок - {best_model_name} ({best_balancing_method})\nAccuracy: {final_accuracy:.4f}')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

            print(f"\n ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ С БАЛАНСИРОВКОЙ:")
            print(f"Лучшая модель: {best_model_name}")
            print(f"Лучший метод балансировки: {best_balancing_method}")
            print(f"Точность: {final_accuracy:.4f}")
            print(f"Улучшение относительно предыдущего результата: +{(final_accuracy - 0.4448) * 100:.2f}%")
            print(f"Количество классов: {len(class_names)}")
            print(f"Количество признаков: {len(feature_columns)}")

        else:
            print("❌ Не удалось обучить ни одну ансамблевую модель")

    else:
        print("❌ Не удалось загрузить данные из JSON файла")