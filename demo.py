"""
Демонстрация работы KazakhErrorGenerator.

Запуск:
    python demo.py
    python demo.py --text "Бүгін ауа райы өте жақсы."
    python demo.py --file clean_texts.txt --output corpus.jsonl --samples 3
"""

import argparse
import json
from kaz_error_generator import KazakhErrorGenerator, CorpusBuilder


# ── Примеры предложений для демо ──────────────────────────────────────────────

DEMO_SENTENCES = [
    "Бүгін ауа райы өте жақсы, күн ашық.",
    "Мен мектепке барамын және сабақ оқимын.",
    "Қазақстан — орталық Азиядағы үлкен мемлекет.",
    "Ол кітапты оқып болғаннан кейін ұйықтап кетті.",
    "Алматы қаласында көптеген тарихи орындар бар.",
    "Жақсы оқушы болу үшін күн сайын жаттығу керек.",
    "Биыл жаз маусымы ерте басталды деп айтады.",
    "Ана тілімізді сақтау — біздің парызымыз.",
]


def demo_single(text: str, n: int = 5, seed: int = 42):
    """Показать N вариантов ошибок для одного предложения."""
    print("\n" + "═" * 60)
    print("ДЕМО: Генерация ошибок для одного предложения")
    print("═" * 60)
    print(f"\nИсходный текст:\n  {text}\n")

    gen = KazakhErrorGenerator(token_error_rate=0.3, seed=seed)

    for i in range(n):
        result = gen.generate(text)
        print(f"Вариант {i + 1}:")
        print(f"  Искажённый: {result.corrupted}")
        if result.annotations:
            for ann in result.annotations:
                print(f"    [{ann.error_type} / {ann.error_subtype}]  "
                      f"'{ann.original}' → '{ann.corrupted}'")
        else:
            print("    (ошибок не внесено)")
        print()


def demo_batch(sentences: list):
    """Показать результаты для набора предложений."""
    print("\n" + "═" * 60)
    print("ДЕМО: Пакетная генерация")
    print("═" * 60)

    gen = KazakhErrorGenerator(token_error_rate=0.20)
    results = gen.generate_batch(sentences)

    total_errors = 0
    from collections import Counter
    type_counter = Counter()

    for result in results:
        changed = result.original != result.corrupted
        marker = "✗" if changed else "✓"
        print(f"\n{marker} Исходный:   {result.original}")
        if changed:
            print(f"  Искажённый: {result.corrupted}")
            for ann in result.annotations:
                print(f"    [{ann.error_type}] '{ann.original}' → '{ann.corrupted}'")
                type_counter[ann.error_type] += 1
                total_errors += 1

    print("\n" + "─" * 40)
    print(f"Всего предложений:  {len(results)}")
    print(f"Всего ошибок:       {total_errors}")
    print("Ошибки по типам:")
    for etype, count in type_counter.most_common():
        print(f"  {etype:<20} {count}")


def demo_corpus(input_file: str, output_file: str, samples: int = 2):
    """Построить мини-корпус из файла."""
    print("\n" + "═" * 60)
    print("ДЕМО: Построение корпуса")
    print("═" * 60)

    gen = KazakhErrorGenerator(token_error_rate=0.15)
    builder = CorpusBuilder(generator=gen, samples_per_text=samples)

    summary = builder.build_from_file(input_file, output_file, verbose=True)

    print(f"\nПервые 3 записи из {output_file}:")
    records = builder.load_corpus(output_file)
    for rec in records[:3]:
        print(json.dumps(rec, ensure_ascii=False, indent=2))


def demo_error_types():
    """Показать каждый тип ошибки на конкретном примере."""
    print("\n" + "═" * 60)
    print("ДЕМО: Типы ошибок")
    print("═" * 60)

    text = "Қазақстанның астанасы — Астана қаласы."

    # special_chars
    gen = KazakhErrorGenerator(
        token_error_rate=1.0,
        error_type_weights={'special_chars': 1.0, 'typos': 0, 'latin_mix': 0,
                             'punctuation': 0, 'spaces': 0}
    )
    r = gen.generate(text)
    print(f"\n[special_chars] Казахские буквы → русские")
    print(f"  Исходный:   {text}")
    print(f"  Искажённый: {r.corrupted}")

    # latin_mix
    gen2 = KazakhErrorGenerator(
        token_error_rate=1.0,
        error_type_weights={'special_chars': 0, 'typos': 0, 'latin_mix': 1.0,
                             'punctuation': 0, 'spaces': 0}
    )
    r2 = gen2.generate(text)
    print(f"\n[latin_mix] Кириллица → латинский омоглиф")
    print(f"  Исходный:   {text}")
    print(f"  Искажённый: {r2.corrupted}")

    # typos
    gen3 = KazakhErrorGenerator(
        token_error_rate=1.0,
        error_type_weights={'special_chars': 0, 'typos': 1.0, 'latin_mix': 0,
                             'punctuation': 0, 'spaces': 0}
    )
    r3 = gen3.generate(text)
    print(f"\n[typos] Опечатки (удвоение/перестановка/пропуск/вставка)")
    print(f"  Исходный:   {text}")
    print(f"  Искажённый: {r3.corrupted}")
    for ann in r3.annotations:
        print(f"    {ann.error_subtype}: '{ann.original}' → '{ann.corrupted}'")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Казахский генератор синтетических ошибок')
    parser.add_argument('--text', type=str, help='Одно предложение для обработки')
    parser.add_argument('--file', type=str, help='Входной файл с чистыми текстами')
    parser.add_argument('--output', type=str, default='corpus.jsonl',
                        help='Выходной файл корпуса (JSONL)')
    parser.add_argument('--samples', type=int, default=2,
                        help='Число вариантов с ошибками на предложение')
    parser.add_argument('--types', action='store_true',
                        help='Показать демо всех типов ошибок')
    args = parser.parse_args()

    if args.types:
        demo_error_types()
    elif args.text:
        demo_single(args.text)
    elif args.file:
        demo_corpus(args.file, args.output, args.samples)
    else:
        # Дефолтный демо-режим
        demo_single(DEMO_SENTENCES[0])
        demo_batch(DEMO_SENTENCES)
        demo_error_types()


if __name__ == '__main__':
    main()
