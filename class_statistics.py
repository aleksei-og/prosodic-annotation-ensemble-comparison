import json
from collections import Counter


def count_labels(json_file_path):
    # Загружаем JSON файл
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Счетчик для всех меток
    label_counter = Counter()

    # Проходим по всем задачам
    for task in data:
        # Проходим по всем аннотациям в задаче
        for annotation in task.get('annotations', []):
            # Проходим по всем результатам в аннотации
            for result in annotation.get('result', []):
                # Получаем метки из поля 'labels'
                labels = result.get('value', {}).get('labels', [])
                # Добавляем метки в счетчик
                label_counter.update(labels)

    return label_counter


def main():
    # Укажите путь к вашему JSON файлу
    json_file_path = 'project-1-at-2025-05-13-11-10-34463d27.json'

    try:
        # Получаем счетчик меток
        label_counts = count_labels(json_file_path)

        # Выводим результаты
        print("Количество экземпляров всех классов:")
        print("-" * 40)

        # Сортируем по убыванию количества
        for label, count in label_counts.most_common():
            print(f"{label}: {count}")

        # Итоговая статистика
        print("-" * 40)
        print(f"Всего уникальных классов: {len(label_counts)}")
        print(f"Всего экземпляров: {sum(label_counts.values())}")

    except FileNotFoundError:
        print(f"Файл {json_file_path} не найден.")
    except json.JSONDecodeError:
        print("Ошибка при чтении JSON файла.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()