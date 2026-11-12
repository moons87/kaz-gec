"""
evaluate_corpus.py — Анализ сгенерированного корпуса ошибок.

Выводит:
  - Общая статистика (пары, ошибки, длины)
  - Распределение типов ошибок
  - Примеры по каждому типу
  - Оценка качества (насколько ошибки реалистичны)
  - График (если установлен matplotlib)

Запуск:
  python evaluate_corpus.py data/corpus.jsonl
  python evaluate_corpus.py data/corpus.jsonl --examples 5 --no-plot
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


# ── Загрузка ──────────────────────────────────────────────────────────────────

def load_corpus(path: str) -> list:
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Метрики ───────────────────────────────────────────────────────────────────

def compute_stats(records: list) -> dict:
    total_pairs = len(records)
    total_errors = 0
    type_counter = Counter()
    subtype_counter = Counter()
    orig_lengths = []
    changed_count = 0
    errors_per_pair = []

    for rec in records:
        orig = rec['original']
        corr = rec['corrupted']
        anns = rec.get('annotations', [])

        orig_lengths.append(len(orig.split()))

        if orig != corr:
            changed_count += 1

        n_errors = len(anns)
        errors_per_pair.append(n_errors)
        total_errors += n_errors

        for ann in anns:
            type_counter[ann['error_type']] += 1
            subtype_counter[ann['error_subtype']] += 1

    avg_len = sum(orig_lengths) / len(orig_lengths) if orig_lengths else 0
    avg_errors = sum(errors_per_pair) / len(errors_per_pair) if errors_per_pair else 0

    return {
        'total_pairs': total_pairs,
        'changed_pairs': changed_count,
        'unchanged_pairs': total_pairs - changed_count,
        'total_errors': total_errors,
        'avg_errors_per_pair': avg_errors,
        'avg_words_per_sent': avg_len,
        'type_counter': type_counter,
        'subtype_counter': subtype_counter,
        'errors_per_pair_dist': Counter(errors_per_pair),
    }


# ── Вывод ─────────────────────────────────────────────────────────────────────

def print_stats(stats: dict):
    total = stats['total_pairs']
    changed = stats['changed_pairs']

    print("\n" + "═" * 58)
    print("  АНАЛИЗ КОРПУСА")
    print("═" * 58)

    print(f"\n  Всего пар:              {total:>8,}")
    print(f"  С ошибками:             {changed:>8,}  ({changed/total*100:.1f}%)")
    print(f"  Без изменений:          {stats['unchanged_pairs']:>8,}  ({stats['unchanged_pairs']/total*100:.1f}%)")
    print(f"  Всего ошибок:           {stats['total_errors']:>8,}")
    print(f"  Ошибок на пару:         {stats['avg_errors_per_pair']:>8.2f}")
    print(f"  Слов в предложении:     {stats['avg_words_per_sent']:>8.1f}")

    # Распределение ошибок по типам
    print(f"\n{'─'*58}")
    print("  ТИПЫ ОШИБОК")
    print(f"{'─'*58}")
    total_err = stats['total_errors']
    type_labels = {
        'special_chars': 'Казахские буквы (ң→н, қ→к...)',
        'typos':         'Опечатки (двойная/пропуск/...)',
        'punctuation':   'Пунктуация',
        'latin_mix':     'Латинско-кирилл. смешение',
        'spaces':        'Пробелы',
    }
    for etype, count in stats['type_counter'].most_common():
        label = type_labels.get(etype, etype)
        bar = '█' * int(count / max(stats['type_counter'].values()) * 30)
        pct = count / total_err * 100 if total_err else 0
        print(f"  {label:<32} {count:>5,}  {pct:>5.1f}%  {bar}")

    # Топ подтипов
    print(f"\n{'─'*58}")
    print("  ТОП-10 ПОДТИПОВ ОШИБОК")
    print(f"{'─'*58}")
    for subtype, count in stats['subtype_counter'].most_common(10):
        pct = count / total_err * 100 if total_err else 0
        print(f"  {subtype:<35} {count:>5,}  {pct:>5.1f}%")

    # Распределение числа ошибок на пару
    print(f"\n{'─'*58}")
    print("  ОШИБОК НА ПАРУ")
    print(f"{'─'*58}")
    for n_err in sorted(stats['errors_per_pair_dist'].keys()):
        count = stats['errors_per_pair_dist'][n_err]
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {n_err} ошибок:  {count:>6,} пар  ({pct:>5.1f}%)  {bar}")

    print("═" * 58)


def print_examples(records: list, n_per_type: int = 3):
    """Показать примеры по каждому типу ошибок."""
    by_type = defaultdict(list)
    for rec in records:
        for ann in rec.get('annotations', []):
            by_type[ann['error_type']].append((rec, ann))

    print(f"\n{'═'*58}")
    print("  ПРИМЕРЫ ПО ТИПАМ")
    print(f"{'═'*58}")

    for etype, pairs in by_type.items():
        print(f"\n  [{etype.upper()}]")
        seen = set()
        count = 0
        for rec, ann in pairs:
            key = (ann['original'], ann['corrupted'])
            if key in seen:
                continue
            seen.add(key)
            print(f"    '{ann['original']}' → '{ann['corrupted']}'  ({ann['error_subtype']})")
            print(f"    Предложение: {rec['original'][:70]}...")
            print(f"    Искажено:   {rec['corrupted'][:70]}...")
            print()
            count += 1
            if count >= n_per_type:
                break


# ── График ────────────────────────────────────────────────────────────────────

def plot_stats(stats: dict, output_path: str = 'data/corpus_stats.png'):
    if not PLOT_AVAILABLE:
        print("\n  (matplotlib не установлен — график пропущен)")
        print("  Установите: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Анализ корпуса казахских ошибок', fontsize=14, fontweight='bold')

    # 1. Типы ошибок — pie chart
    ax1 = axes[0]
    labels = list(stats['type_counter'].keys())
    sizes = list(stats['type_counter'].values())
    short_labels = {
        'special_chars': 'Спец. буквы',
        'typos': 'Опечатки',
        'punctuation': 'Пунктуация',
        'latin_mix': 'Лат/Кир',
        'spaces': 'Пробелы',
    }
    labels_short = [short_labels.get(l, l) for l in labels]
    ax1.pie(sizes, labels=labels_short, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Типы ошибок')

    # 2. Подтипы — горизонтальный bar chart
    ax2 = axes[1]
    top_subtypes = stats['subtype_counter'].most_common(10)
    subtypes = [s[0] for s in top_subtypes][::-1]
    counts = [s[1] for s in top_subtypes][::-1]
    ax2.barh(subtypes, counts, color='steelblue')
    ax2.set_title('Топ-10 подтипов')
    ax2.set_xlabel('Количество')

    # 3. Распределение ошибок на пару
    ax3 = axes[2]
    dist = stats['errors_per_pair_dist']
    x = sorted(dist.keys())
    y = [dist[k] for k in x]
    ax3.bar([str(i) for i in x], y, color='coral')
    ax3.set_title('Ошибок на пару')
    ax3.set_xlabel('Количество ошибок')
    ax3.set_ylabel('Пар')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  График сохранён: {output_path}")
    plt.show()


# ── Пригодность для обучения ──────────────────────────────────────────────────

def check_training_readiness(stats: dict):
    total = stats['total_pairs']
    changed_pct = stats['changed_pairs'] / total * 100 if total else 0

    print(f"\n{'═'*58}")
    print("  ГОТОВНОСТЬ К ОБУЧЕНИЮ GECToR")
    print(f"{'═'*58}")

    checks = [
        (total >= 5000,    f"Размер корпуса: {total:,} пар",
         "Нужно минимум 5 000 — запустите с --limit 30000"),
        (total >= 20000,   f"Рекомендуемый размер: {total:,} пар",
         "Рекомендуется 20 000+ для стабильного обучения"),
        (changed_pct >= 70, f"Доля изменённых пар: {changed_pct:.1f}%",
         "Рекомендуется 70%+ — увеличьте token_error_rate"),
        (len(stats['type_counter']) >= 3, f"Разнообразие типов: {len(stats['type_counter'])}",
         "Нужно минимум 3 типа ошибок"),
    ]

    all_ok = True
    for ok, label, advice in checks:
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {label}")
        if not ok:
            print(f"      → {advice}")
            all_ok = False

    if all_ok:
        print(f"\n  ✓ Корпус готов к обучению!")
    else:
        print(f"\n  ✗ Доработайте корпус перед обучением")

    print("═" * 58)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Анализ корпуса казахских ошибок')
    parser.add_argument('corpus', help='Путь к JSONL файлу корпуса')
    parser.add_argument('--examples', type=int, default=2,
                        help='Примеров на тип ошибки (по умолчанию: 2)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Не строить график')
    parser.add_argument('--no-examples', action='store_true',
                        help='Не показывать примеры')
    args = parser.parse_args()

    path = args.corpus
    if not Path(path).exists():
        print(f"Файл не найден: {path}")
        sys.exit(1)

    print(f"\nЗагружаем корпус: {path}")
    records = load_corpus(path)
    print(f"Загружено записей: {len(records):,}")

    stats = compute_stats(records)
    print_stats(stats)
    check_training_readiness(stats)

    if not args.no_examples:
        print_examples(records, n_per_type=args.examples)

    if not args.no_plot:
        plot_stats(stats)


if __name__ == '__main__':
    main()
