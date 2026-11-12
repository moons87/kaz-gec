"""
pipeline.py — Полный пайплайн генерации корпуса ошибок.

Шаги:
  1. Загрузить казахские предложения из MDBKD или KazakhTTS (HuggingFace)
  2. Применить KazakhErrorGenerator к каждому предложению
  3. Сохранить корпус в JSONL формат

Запуск (примеры):
  # Быстрый тест: 1000 предложений, 2 варианта каждого
  python pipeline.py --limit 1000 --samples 2

  # Полный запуск: 100 000 предложений
  python pipeline.py --limit 100000 --samples 3 --output data/corpus_100k.jsonl

  # Использовать KazakhTTS вместо MDBKD
  python pipeline.py --dataset kazakh-tts --limit 50000

  # Диагностика KazakhTTS (узнать поля датасета)
  python pipeline.py --dataset kazakh-tts --diagnose

  # Использовать уже скачанный файл
  python pipeline.py --input data/clean.txt --samples 2

  # Только скачать тексты (без генерации ошибок)
  python pipeline.py --download-only --limit 50000 --output data/clean.txt
"""

import argparse
import time
from pathlib import Path

from kaz_error_generator import KazakhErrorGenerator, CorpusBuilder
from kaz_error_generator.dataset_loader import DatasetLoader, HF_AVAILABLE


def _make_loader(dataset: str, hf_token: str, no_lang_filter: bool) -> DatasetLoader:
    """Создать DatasetLoader в зависимости от выбранного датасета."""
    if dataset == 'kazakh-tts':
        print(f"  Датасет: KazakhTTS ({DatasetLoader.KAZAKH_TTS})")
        return DatasetLoader.from_kazakh_tts(token=hf_token)
    else:
        print(f"  Датасет: MDBKD ({DatasetLoader.MDBKD})")
        return DatasetLoader(
            min_sent_len=15,
            max_sent_len=250,
            min_kaz_chars=1,
            language_field=None if no_lang_filter else 'predicted_language',
            target_language='kaz',
            token=hf_token,
            heartbeat=500,
        )


def run_pipeline(
    limit: int = 5000,
    samples_per_text: int = 2,
    token_error_rate: float = 0.15,
    output: str = 'data/corpus.jsonl',
    input_file: str = None,
    download_only: bool = False,
    seed: int = 42,
    verbose: bool = True,
    hf_token: str = None,
    no_lang_filter: bool = False,
    dataset: str = 'mdbkd',
):
    Path('data').mkdir(exist_ok=True)

    # ── Шаг 1: Получить чистые тексты ────────────────────────────────────────
    if input_file:
        print(f"\n[1/2] Загружаем тексты из файла: {input_file}")
        clean_file = input_file
    else:
        if not HF_AVAILABLE:
            print("ОШИБКА: библиотека 'datasets' не установлена.")
            print("Установите: pip install datasets")
            return

        clean_file = 'data/clean_kazakh.txt'
        print(f"\n[1/2] Загружаем {limit} предложений из HuggingFace...")
        print("      Каждые 500 записей — строка статуса\n")

        t0 = time.time()
        loader = _make_loader(dataset, hf_token, no_lang_filter)
        n = loader.save_to_file(clean_file, limit=limit, verbose=verbose)
        elapsed = time.time() - t0
        print(f"\n  Загружено за {elapsed:.1f}с: {n} предложений → {clean_file}")

        if download_only:
            print("\nРежим --download-only: генерация ошибок пропущена.")
            return

    # ── Шаг 2: Генерация ошибок ───────────────────────────────────────────────
    print(f"\n[2/2] Генерируем корпус ошибок...")
    print(f"      error_rate={token_error_rate}, samples={samples_per_text}\n")

    gen = KazakhErrorGenerator(
        token_error_rate=token_error_rate,
        seed=seed,
    )
    builder = CorpusBuilder(
        generator=gen,
        samples_per_text=samples_per_text,
        min_length=15,
    )

    t0 = time.time()
    summary = builder.build_from_file(
        input_path=clean_file,
        output_path=output,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    print(f"\nВремя генерации: {elapsed:.1f}с")
    print(f"Корпус сохранён: {output}")

    # ── Показать несколько примеров ───────────────────────────────────────────
    print("\n── Примеры из корпуса ──────────────────────────────────")
    import json
    with open(output, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            rec = json.loads(line)
            print(f"\n  Исходный:   {rec['original']}")
            print(f"  Искажённый: {rec['corrupted']}")
            for ann in rec['annotations']:
                print(f"    [{ann['error_type']}] "
                      f"'{ann['original']}' → '{ann['corrupted']}'")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Пайплайн генерации казахского корпуса ошибок'
    )
    parser.add_argument('--limit', type=int, default=5000,
                        help='Количество предложений из датасета (по умолчанию: 5000)')
    parser.add_argument('--samples', type=int, default=2,
                        help='Вариантов с ошибками на предложение (по умолчанию: 2)')
    parser.add_argument('--error-rate', type=float, default=0.15,
                        help='Вероятность ошибки на токен (по умолчанию: 0.15)')
    parser.add_argument('--output', type=str, default='data/corpus.jsonl',
                        help='Выходной JSONL файл (по умолчанию: data/corpus.jsonl)')
    parser.add_argument('--input', type=str, default=None,
                        help='Использовать готовый файл с текстами (пропустить загрузку)')
    parser.add_argument('--download-only', action='store_true',
                        help='Только скачать тексты, без генерации ошибок')
    parser.add_argument('--seed', type=int, default=42,
                        help='Зерно генератора (для воспроизводимости)')
    parser.add_argument('--quiet', action='store_true',
                        help='Меньше вывода')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace токен (hf_xxx...)')
    parser.add_argument('--diagnose', action='store_true',
                        help='Проверить структуру датасета перед запуском')
    parser.add_argument('--no-lang-filter', action='store_true',
                        help='Отключить фильтр по языку (взять все записи)')
    parser.add_argument('--dataset', type=str, default='mdbkd',
                        choices=['mdbkd', 'kazakh-tts'],
                        help='Источник текстов: mdbkd (по умолчанию) или kazakh-tts')
    args = parser.parse_args()

    if args.diagnose:
        loader = _make_loader(args.dataset, args.token, args.no_lang_filter)
        loader.diagnose(n=10)
        return

    run_pipeline(
        limit=args.limit,
        samples_per_text=args.samples,
        token_error_rate=args.error_rate,
        output=args.output,
        input_file=args.input,
        download_only=args.download_only,
        seed=args.seed,
        verbose=not args.quiet,
        hf_token=args.token,
        no_lang_filter=args.no_lang_filter,
        dataset=args.dataset,
    )


if __name__ == '__main__':
    main()
