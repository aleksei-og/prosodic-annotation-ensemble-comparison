import json
import os
from collections import Counter


def analyze_audio_files(json_file_path):
    # Загружаем JSON файл
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Множество для хранения уникальных имен аудиофайлов
    unique_audio_files = set()
    file_extensions = Counter()

    print("Детальная информация об аудиозаписях:")
    print("=" * 60)

    # Проходим по всем задачам
    for i, task in enumerate(data, 1):
        task_id = task.get('id', 'N/A')
        audio_source = None
        audio_file = None

        # Пробуем получить из file_upload
        if 'file_upload' in task:
            audio_file = task['file_upload']
            audio_source = 'file_upload'

        # Пробуем получить из data->audio
        elif 'data' in task and 'audio' in task['data']:
            audio_path = task['data']['audio']
            audio_file = os.path.basename(audio_path)
            audio_source = 'data.audio'

        if audio_file:
            unique_audio_files.add(audio_file)
            # Анализируем расширение файла
            _, ext = os.path.splitext(audio_file)
            file_extensions[ext.lower()] += 1

            print(f"Задача {i:2} (ID: {task_id}): {audio_file}")
            print(f"         Источник: {audio_source}")
        else:
            print(f"Задача {i:2} (ID: {task_id}): Аудиофайл не найден")

    return unique_audio_files, file_extensions


def main():
    # Укажите путь к вашему JSON файлу
    json_file_path = 'project-1-at-2025-05-13-11-10-34463d27.json'

    try:
        # Получаем информацию об аудиофайлах
        unique_audio_files, file_extensions = analyze_audio_files(json_file_path)

        # Выводим итоговую статистику
        print("\n" + "=" * 60)
        print("ИТОГОВАЯ СТАТИСТИКА:")
        print("-" * 60)

        print("\nУникальные аудиозаписи:")
        for i, audio_file in enumerate(sorted(unique_audio_files), 1):
            print(f"{i:2}. {audio_file}")

        print(f"\nВсего уникальных аудиозаписей: {len(unique_audio_files)}")
        print(f"Всего задач в JSON: {len(data) if 'data' in locals() else 'N/A'}")

        print("\nРаспределение по расширениям файлов:")
        for ext, count in file_extensions.most_common():
            print(f"  {ext or 'без расширения'}: {count}")

    except FileNotFoundError:
        print(f"Файл {json_file_path} не найден.")
    except json.JSONDecodeError:
        print("Ошибка при чтении JSON файла.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()