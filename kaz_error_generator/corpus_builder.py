"""
CorpusBuilder — пакетная генерация корпуса ошибок из чистого текста.

Входные данные:  файл с чистыми казахскими предложениями (по одному на строку)
                 или список строк
Выходные данные: JSONL-файл с парами (original, corrupted, annotations)
                 + статистика по типам ошибок
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Iterator
from collections import Counter

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .generator import KazakhErrorGenerator, GenerationResult


class CorpusBuilder:
    """
    Строит параллельный корпус ошибок для обучения GEC-модели.

    Параметры:
        generator         — экземпляр KazakhErrorGenerator
        samples_per_text  — сколько вариантов с ошибками создать для каждого предложения
        min_length        — минимальная длина предложения в символах (короткие пропускаем)
        max_length        — максимальная длина (длинные обрезаем)
    """

    def __init__(
        self,
        generator: Optional[KazakhErrorGenerator] = None,
        samples_per_text: int = 1,
        min_length: int = 10,
        max_length: int = 512,
    ):
        self.generator = generator or KazakhErrorGenerator()
        self.samples_per_text = samples_per_text
        self.min_length = min_length
        self.max_length = max_length

    # ── Основные методы ───────────────────────────────────────────────────────

    def build_from_file(
        self,
        input_path: str,
        output_path: str,
        limit: Optional[int] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Построить корпус из файла с чистым текстом.

        Каждая строка файла считается одним предложением.
        Результат сохраняется в JSONL формате.

        Возвращает статистику генерации.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Входной файл не найден: {input_path}")

        lines = self._read_lines(input_path)
        if limit:
            lines = lines[:limit]

        return self._build(lines, output_path, verbose)

    def build_from_texts(
        self,
        texts: List[str],
        output_path: str,
        verbose: bool = True,
    ) -> dict:
        """Построить корпус из списка строк."""
        return self._build(texts, Path(output_path), verbose)

    # ── Внутренняя логика ─────────────────────────────────────────────────────

    def _read_lines(self, path: Path) -> List[str]:
        with open(path, encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def _build(self, texts: List[str], output_path: Path, verbose: bool) -> dict:
        stats = Counter()
        total_generated = 0
        skipped = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)

        bar = None
        if verbose and TQDM_AVAILABLE:
            bar = tqdm(
                total=len(texts),
                desc="Генерация ошибок",
                unit=" пред",
                bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] пар: {postfix}",
                dynamic_ncols=True,
            )

        with open(output_path, 'w', encoding='utf-8') as out:
            for i, text in enumerate(texts):
                if len(text) < self.min_length:
                    skipped += 1
                    if bar:
                        bar.update(1)
                    continue
                text = text[:self.max_length]

                for _ in range(self.samples_per_text):
                    result = self.generator.generate(text)
                    if result.original == result.corrupted:
                        continue
                    out.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')
                    total_generated += 1
                    for ann in result.annotations:
                        stats[ann.error_type] += 1

                if bar:
                    bar.set_postfix_str(str(total_generated))
                    bar.update(1)
                elif verbose and (i + 1) % 500 == 0:
                    print(f"  Обработано: {i + 1}/{len(texts)}, пар: {total_generated}")

        if bar:
            bar.close()

        summary = {
            'input_sentences': len(texts),
            'skipped': skipped,
            'generated_pairs': total_generated,
            'errors_by_type': dict(stats),
            'output_file': str(output_path),
        }

        if verbose:
            self._print_summary(summary)

        return summary

    def _print_summary(self, summary: dict):
        print("\n" + "─" * 50)
        print("Статистика генерации корпуса")
        print("─" * 50)
        print(f"  Входных предложений:  {summary['input_sentences']}")
        print(f"  Пропущено (короткие): {summary['skipped']}")
        print(f"  Сгенерировано пар:    {summary['generated_pairs']}")
        print()
        print("  Ошибки по типам:")
        total_errors = sum(summary['errors_by_type'].values())
        for etype, count in sorted(summary['errors_by_type'].items(),
                                   key=lambda x: -x[1]):
            pct = count / total_errors * 100 if total_errors else 0
            print(f"    {etype:<20} {count:>6}  ({pct:.1f}%)")
        print("─" * 50)

    # ── Утилиты ───────────────────────────────────────────────────────────────

    def load_corpus(self, path: str) -> List[dict]:
        """Загрузить сгенерированный корпус из JSONL-файла."""
        records = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def get_stats(self, path: str) -> dict:
        """Вычислить статистику по существующему корпусу."""
        records = self.load_corpus(path)
        stats = Counter()
        for rec in records:
            for ann in rec.get('annotations', []):
                stats[ann['error_type']] += 1
        return {
            'total_pairs': len(records),
            'total_errors': sum(stats.values()),
            'errors_by_type': dict(stats),
        }
