import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import cross_val_score
from collections import Counter
import warnings
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')


def load_data_from_json(json_file_path):
    """Загружает и преобразует данные из JSON файла в DataFrame"""
    print(f"Загрузка данных из JSON файла: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_annotations = []
    for task in data:
        task_id = task['id']
        file_name = task['file_upload']
        for annotation in task['annotations']:
            for result in annotation['result']:
                if result['type'] == 'labels':
                    segment_info = {
                        'task_id': task_id,
                        'file_name': file_name,
                        'start_time': result['value']['start'],
                        'end_time': result['value']['end'],
                        'duration': result['value']['end'] - result['value']['start'],
                        'label': result['value']['labels'][0],
                        'channel': result['value']['channel'],
                        'original_length': result['original_length']
                    }
                    all_annotations.append(segment_info)
    df = pd.DataFrame(all_annotations)
    print(f"Загружено {len(df)} сегментов аудио")
    return df


def extract_advanced_audio_features(df):
    """Извлекает расширенные признаки из временных сегментов аудио"""
    print("\nИзвлечение РАСШИРЕННЫХ признаков из аудио сегментов...")
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
    features_df['silence_before'] = features_df['start_time'] - features_df['prev_end_time']
    features_df['silence_after'] = features_df.groupby('file_name')['start_time'].shift(-1) - features_df['end_time']
    features_df['silence_before'] = features_df['silence_before'].fillna(0)
    features_df['silence_after'] = features_df['silence_after'].fillna(0)
    features_df['prev_duration'] = features_df['prev_duration'].fillna(features_df['duration'])
    features_df['next_duration'] = features_df['next_duration'].fillna(features_df['duration'])
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
    features_df['rolling_duration_mean'] = features_df.groupby('file_name')['duration'].rolling(window=window_size,
                                                                                                min_periods=1).mean().reset_index(
        drop=True)
    features_df['rolling_duration_std'] = features_df.groupby('file_name')['duration'].rolling(window=window_size,
                                                                                               min_periods=1).std().reset_index(
        drop=True)

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
    columns_to_drop = ['prev_end_time']
    features_df = features_df.drop(columns=[col for col in columns_to_drop if col in features_df.columns])
    features_df = features_df.fillna(0)
    return features_df


def create_improved_features(df):
    """Создает улучшенные признаки на основе анализа важности"""
    print("\n СОЗДАНИЕ УЛУЧШЕННЫХ ПРИЗНАКОВ НА ОСНОВЕ АНАЛИЗА ВАЖНОСТИ")
    features_df = df.copy()

    # 1. УСИЛЕНИЕ ТОП-ПРИЗНАКОВ
    print("1. Усиление топ-признаков...")
    features_df['prev_duration_x_acceleration'] = features_df['prev_duration'] * features_df['acceleration']
    features_df['duration_ratio_prev_x_rolling_std'] = features_df['duration_ratio_prev'] * features_df[
        'rolling_duration_std']
    features_df['acceleration_x_rolling_mean'] = features_df['acceleration'] * features_df['rolling_duration_mean']

    # 2. НОВЫЕ ПРИЗНАКИ НА ОСНОВЕ ДЛИТЕЛЬНОСТИ
    print("2. Создание новых признаков длительности...")
    features_df['rolling_duration_skew'] = features_df.groupby('file_name')['duration'].rolling(window=3,
                                                                                                min_periods=1).skew().reset_index(
        drop=True)
    features_df['rolling_duration_kurt'] = features_df.groupby('file_name')['duration'].rolling(window=3,
                                                                                                min_periods=1).kurt().reset_index(
        drop=True)
    features_df['duration_momentum'] = features_df['duration_change_prev'] - features_df.groupby('file_name')[
        'duration_change_prev'].shift(1)
    features_df['duration_volatility'] = features_df.groupby('file_name')['duration_change_prev'].rolling(window=3,
                                                                                                          min_periods=1).std().reset_index(
        drop=True)

    # 3. ПРИЗНАКИ РИТМА И ТЕМПА (усиленные)
    print("3. Улучшенные признаки ритма...")
    features_df['speech_consistency'] = features_df['file_duration_std'] / (features_df['file_duration_mean'] + 0.001)
    features_df['pause_pattern'] = (features_df['silence_before'] + features_df['silence_after']) / (
                features_df['duration'] + 0.001)
    features_df['rhythm_complexity'] = features_df['rolling_duration_std'] * features_df['total_segments_in_file']

    # 4. ПРИЗНАКИ ПОЗИЦИИ И СТРУКТУРЫ
    print("4. Признаки позиции и структуры...")
    features_df['position_quadratic'] = features_df['position_in_file'] ** 2
    features_df['structural_importance'] = features_df['is_first_segment'] * 2 + features_df['is_last_segment'] * 1.5

    # 5. ВЗАИМОДЕЙСТВИЯ С КАТЕГОРИАЛЬНЫМИ ПРИЗНАКАМИ
    print("5. Взаимодействия с категориальными признаками...")
    features_df['early_short'] = features_df['is_early'] * features_df['is_very_short']
    features_df['late_long'] = features_df['is_late'] * features_df['is_long']
    features_df['middle_medium'] = features_df['is_middle'] * features_df['is_medium']

    # 6. ПРИЗНАКИ НА ОСНОВЕ СОСЕДЕЙ (расширенные)
    print("6. Расширенные признаки соседей...")
    features_df['neighbor_duration_avg'] = (features_df['prev_duration'] + features_df['next_duration']) / 2
    features_df['duration_trend'] = (features_df['next_duration'] - features_df['prev_duration']) / (
                features_df['prev_duration'] + 0.001)
    features_df['stability_score'] = 1 / (features_df['rolling_duration_std'] + 0.001)

    # 7. ВРЕМЕННЫЕ ПАТТЕРНЫ
    print("7. Временные паттерны...")
    features_df['time_pattern_sin2'] = np.sin(4 * np.pi * features_df['time_ratio'])
    features_df['time_pattern_cos2'] = np.cos(4 * np.pi * features_df['time_ratio'])
    features_df['seasonal_interaction'] = features_df['time_sin'] * features_df['duration']

    features_df = features_df.fillna(0)
    print(f" Создано {len(features_df.columns) - len(df.columns)} новых признаков")
    return features_df


def select_best_features_balanced(X, y, feature_names, top_k=30):
    """Улучшенный выбор признаков с балансом разных типов"""

    print(f"\n УЛУЧШЕННЫЙ ВЫБОР ПРИЗНАКОВ (топ-{top_k})")

    # Метод 1: Feature Importance от Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    rf_importance = rf.feature_importances_

    # Метод 2: Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)

    # Комбинированная оценка
    combined_scores = (rf_importance * 0.7 + mi_scores * 0.3)  # Больше вес feature importance

    # Создаем DataFrame с оценками
    feature_scores = pd.DataFrame({
        'feature': feature_names,
        'rf_importance': rf_importance,
        'mi_score': mi_scores,
        'combined_score': combined_scores
    }).sort_values('combined_score', ascending=False)

    # Выбираем топ-K признаков
    selected_features = feature_scores.head(top_k)['feature'].tolist()
    selected_scores = feature_scores.head(top_k)['combined_score'].tolist()

    print("Топ-15 лучших признаков:")
    for i, (feature, score) in enumerate(zip(selected_features[:15], selected_scores[:15])):
        print(f"  {i + 1:2d}. {feature}: {score:.4f}")

    # Создаем отфильтрованные данные
    selected_indices = [feature_names.index(f) for f in selected_features]
    X_selected = X[:, selected_indices]

    return selected_features, X_selected


def optimized_ensemble_improved(X_train, X_test, y_train, y_test, feature_names, class_names):
    """Оптимизированный ансамбль с улучшенными моделями"""

    print("\n" + "=" * 80)
    print(" УЛУЧШЕННЫЙ АНСАМБЛЬ С БАЛАНСИРОВАННЫМИ ПРИЗНАКАМИ")
    print("=" * 80)

    # Отбираем лучшие признаки
    selected_features, X_train_selected = select_best_features_balanced(X_train, y_train, feature_names, top_k=35)
    X_test_selected = X_test[:, [feature_names.index(f) for f in selected_features]]

    print(f" Используется {len(selected_features)} признаков из {len(feature_names)}")

    # УЛУЧШЕННЫЕ МОДЕЛИ
    ensemble_models = {
        'Voting Enhanced': VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=42
                )),
                ('et', ExtraTreesClassifier(
                    n_estimators=150,
                    max_depth=20,
                    class_weight='balanced',
                    random_state=42
                )),
                ('knn', KNeighborsClassifier(
                    n_neighbors=7,
                    weights='distance',
                    metric='minkowski'
                ))
            ],
            voting='soft',
            weights=[3, 2, 1]
        ),

        'RF Optimized': RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            max_features='sqrt',
            bootstrap=True
        ),

        'SVM Balanced': SVC(
            kernel='rbf',
            C=0.1,  # Уменьшили C для лучшей обобщающей способности
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        ),

        'Stacking Enhanced': StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
                ('et', ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
            ],
            final_estimator=LogisticRegression(
                class_weight='balanced',
                C=0.1,
                random_state=42,
                max_iter=1000
            ),
            cv=3
        ),

        'AdaBoost Tuned': AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.05,  # Уменьшили learning rate
            random_state=42
        )
    }

    results = []

    for name, model in ensemble_models.items():
        print(f"\n Обучение {name}...")
        start_time = time.time()

        try:
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)
            training_time = time.time() - start_time

            # Расширенные метрики
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Кросс-валидация
            cv_scores = cross_val_score(model, X_train_selected, y_train, cv=3, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'F1 Macro': f1_macro,
                'F1 Weighted': f1_weighted,
                'CV Mean': cv_mean,
                'CV Std': cv_std,
                'Training Time': training_time
            })

            print(f"     Точность: {accuracy:.4f}")
            print(f"     F1 Macro: {f1_macro:.4f}, F1 Weighted: {f1_weighted:.4f}")
            print(f"     CV Score: {cv_mean:.4f} ± {cv_std:.4f}")

            if accuracy > 0.43:  # Лучше предыдущего лучшего результата
                print(f"    🎉 УЛУЧШЕНИЕ! Детальный отчет:")
                print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

        except Exception as e:
            print(f"     Ошибка: {e}")
            results.append({
                'Model': name,
                'Accuracy': 0,
                'F1 Macro': 0,
                'F1 Weighted': 0,
                'CV Mean': 0,
                'CV Std': 0,
                'Training Time': 0
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)

    print("\n" + "=" * 80)
    print(" ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 80)
    print(results_df.to_string(index=False))

    return results_df, ensemble_models, selected_features


def analyze_feature_importance_ensemble(best_ensemble_model, feature_names, top_n=20):
    """Анализ важности признаков для ансамблевых моделей"""
    plt.figure(figsize=(12, 8))

    if hasattr(best_ensemble_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_ensemble_model.feature_importances_
        }).sort_values('importance', ascending=True).tail(top_n)

        plt.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
        plt.title(f'Топ-{top_n} самых важных признаков\n({type(best_ensemble_model).__name__})')
        plt.xlabel('Важность признака')

    elif hasattr(best_ensemble_model, 'estimators_'):
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

    else:
        plt.text(0.5, 0.5, 'Важность признаков недоступна\nдля этого типа ансамбля',
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Анализ важности признаков')

    plt.tight_layout()
    plt.show()


# === ОСНОВНАЯ ПРОГРАММА ===
if __name__ == "__main__":
    json_file_path = "project-1-at-2025-05-13-11-10-34463d27.json"
    print(" УЛУЧШЕННАЯ КЛАССИФИКАЦИЯ С БАЛАНСИРОВАННЫМИ ПРИЗНАКАМИ")
    print("=" * 80)

    # Загрузка данных
    raw_data = load_data_from_json(json_file_path)

    if raw_data is not None:
        # Базовые признаки
        features_data = extract_advanced_audio_features(raw_data)

        # УЛУЧШЕННЫЕ ПРИЗНАКИ
        improved_data = create_improved_features(features_data)

        # Фильтрация редких классов
        class_counts = improved_data['label'].value_counts()
        valid_classes = class_counts[class_counts >= 5].index
        filtered_data = improved_data[improved_data['label'].isin(valid_classes)]

        print(f"\n УЛУЧШЕННЫЙ ДАТАСЕТ:")
        print(f"   Образцов: {filtered_data.shape[0]}")
        print(f"   Признаков: {len(filtered_data.columns)}")
        print(f"   Классов: {len(valid_classes)}")

        # Подготовка данных
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

        # Масштабирование
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\n ДАННЫЕ ДЛЯ ОБУЧЕНИЯ:")
        print(f"   Обучающая выборка: {X_train_scaled.shape}")
        print(f"   Тестовая выборка: {X_test_scaled.shape}")
        print(f"   Исходных признаков: {len(feature_columns)}")

        # Улучшенный ансамбль с балансированными признаками
        results_df, ensemble_models, selected_features = optimized_ensemble_improved(
            X_train_scaled, X_test_scaled, y_train, y_test, feature_columns, class_names
        )

        # Анализ лучшей модели
        if len(results_df) > 0 and results_df.iloc[0]['Accuracy'] > 0:
            best_result = results_df.iloc[0]
            best_model_name = best_result['Model']
            best_model = ensemble_models[best_model_name]

            print(f"\n ЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
            print("=" * 50)
            print(f"Точность: {best_result['Accuracy']:.4f}")
            print(f"F1 Weighted: {best_result['F1 Weighted']:.4f}")
            print(f"Улучшение относительно baseline: +{(best_result['Accuracy'] - 0.4331) * 100:.2f}%")

            # Переобучаем лучшую модель на всех данных
            selected_indices = [feature_columns.index(f) for f in selected_features]
            X_train_final = X_train_scaled[:, selected_indices]
            X_test_final = X_test_scaled[:, selected_indices]

            best_model.fit(X_train_final, y_train)
            y_pred_final = best_model.predict(X_test_final)
            final_accuracy = accuracy_score(y_test, y_pred_final)

            print(f"\n ФИНАЛЬНАЯ ТОЧНОСТЬ: {final_accuracy:.4f}")

            # Анализ важности признаков
            if hasattr(best_model, 'feature_importances_') or hasattr(best_model, 'estimators_'):
                print(f"\n🔍 АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ ДЛЯ {best_model_name}:")
                analyze_feature_importance_ensemble(best_model, selected_features, top_n=15)

            # Матрица ошибок
            plt.figure(figsize=(12, 10))
            cm = confusion_matrix(y_test, y_pred_final)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Матрица ошибок - {best_model_name}\nAccuracy: {final_accuracy:.4f}')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

            print(f"\n ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
            print(f"Лучшая модель: {best_model_name}")
            print(f"Финальная точность: {final_accuracy:.4f}")
            print(f"Количество классов: {len(class_names)}")
            print(f"Количество использованных признаков: {len(selected_features)}")

    else:
        print("❌ Не удалось загрузить данные из JSON файла")