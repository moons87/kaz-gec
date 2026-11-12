"""
DatasetLoader — загрузка и фильтрация казахских текстов из HuggingFace.

Поддерживаемые датасеты:
  - kz-transformers/multidomain-kazakh-dataset  (MDBKD, 25 ГБ)
  - issai/KazakhTTS                             (чистые казахские предложения)

Используется потоковая загрузка (streaming=True).

Примеры использования:

  # MDBKD (по умолчанию)
  loader = DatasetLoader()

  # KazakhTTS — сначала запусти diagnose() чтобы узнать точное поле текста
  loader = DatasetLoader.from_kazakh_tts()
  loader.diagnose()  # покажет доступные поля и примеры
"""

import re
import sys
import time
import unicodedata
import warnings
from typing import Iterator, List, Optional

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Казахские специфические буквы (строчные + заглавные, без plain А/а)
_KAZ_CHARS = set('әіңғқөүұһӘІҢҒҚӨҮҰҺ')

# Разбивка на предложения
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[А-ЯЁӘІҢҒҚӨҮҰҺA-Z\d«"—])')


def _contains_kazakh(text: str, min_kaz_chars: int = 2) -> bool:
    return sum(1 for ch in text if ch in _KAZ_CHARS) >= min_kaz_chars


def _split_sentences(text: str) -> List[str]:
    paragraphs = re.split(r'\n{2,}', text)
    sentences = []
    for para in paragraphs:
        para = para.strip()
        if para:
            sentences.extend(s.strip() for s in _SENT_RE.split(para) if s.strip())
    return sentences


def _normalize(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


class DatasetLoader:
    """
    Потоковый загрузчик казахских текстов из HuggingFace.

    Параметры:
        dataset_name      — имя датасета на HuggingFace Hub
        split             — сплит датасета
        text_field        — поле с текстом (проверяется автоматически)
        min_sent_len      — минимальная длина предложения в символах
        max_sent_len      — максимальная длина предложения
        min_kaz_chars     — минимум казахских спецбукв (0 = отключить фильтр)
        language_field    — поле с языком (None = не фильтровать)
        target_language   — целевой язык ('kk')
        token             — HuggingFace токен
        trust_remote_code — передать в load_dataset
        heartbeat         — выводить строку статуса каждые N записей (0 = выкл)
    """

    MDBKD = 'kz-transformers/multidomain-kazakh-dataset'
    KAZAKH_TTS = 'issai/KazakhTTS'

    @classmethod
    def from_kazakh_tts(
        cls,
        text_field: str = 'txt',
        split: str = 'train',
        token: Optional[str] = None,
        **kwargs,
    ) -> 'DatasetLoader':
        """
        Удобный конструктор для KazakhTTS.
        Языковой фильтр отключён — датасет уже весь на казахском.
        Запусти .diagnose() после создания чтобы проверить поля.
        """
        return cls(
            dataset_name=cls.KAZAKH_TTS,
            split=split,
            text_field=text_field,
            language_field=None,   # не нужен — датасет уже на казахском
            target_language=None,
            token=token,
            **kwargs,
        )

    def __init__(
        self,
        dataset_name: str = MDBKD,
        split: str = 'train',
        text_field: str = 'text',
        min_sent_len: int = 15,
        max_sent_len: int = 300,
        min_kaz_chars: int = 1,
        language_field: Optional[str] = 'predicted_language',
        target_language: Optional[str] = 'kaz',
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        heartbeat: int = 500,
    ):
        if not HF_AVAILABLE:
            raise ImportError("Установите: pip install datasets")

        self.dataset_name = dataset_name
        self.split = split
        self.text_field = text_field
        self.min_sent_len = min_sent_len
        self.max_sent_len = max_sent_len
        self.min_kaz_chars = min_kaz_chars
        self.language_field = language_field
        self.target_language = target_language
        self.token = token
        self.trust_remote_code = trust_remote_code
        self.heartbeat = heartbeat

    # ── Диагностика ───────────────────────────────────────────────────────────

    def diagnose(self, n: int = 5):
        """
        Показать первые N записей датасета и проверить поля.
        Используйте перед запуском пайплайна, чтобы убедиться что всё корректно.
        """
        print(f"\n{'─'*60}")
        print(f"ДИАГНОСТИКА: {self.dataset_name}")
        print(f"{'─'*60}")

        dataset = self._load_dataset()
        records = []
        for i, rec in enumerate(dataset):
            if i >= n:
                break
            records.append(rec)

        if not records:
            print("❌ Датасет пуст или недоступен")
            return

        print(f"\n✓ Загружено {len(records)} записей")
        print(f"\nДоступные поля: {list(records[0].keys())}")

        # Проверка text_field
        if self.text_field in records[0]:
            sample = records[0][self.text_field]
            print(f"\n✓ text_field='{self.text_field}' найдено")
            print(f"  Пример: {repr(sample[:150])}...")
        else:
            print(f"\n❌ text_field='{self.text_field}' НЕ найдено!")
            print(f"   Доступные текстовые поля: "
                  f"{[k for k,v in records[0].items() if isinstance(v, str)]}")

        # Проверка language_field
        if self.language_field:
            if self.language_field in records[0]:
                langs = [r.get(self.language_field) for r in records]
                print(f"\n✓ language_field='{self.language_field}' найдено")
                print(f"  Значения в первых {n} записях: {langs}")
                kk_count = sum(1 for l in langs if l == self.target_language)
                print(f"  Казахских ('{self.target_language}'): {kk_count}/{len(langs)}")
                if kk_count == 0:
                    print(f"  ⚠️  Ни одной записи с language='{self.target_language}'!"
                          f" Попробуйте language_field=None для отключения фильтра.")
            else:
                print(f"\n❌ language_field='{self.language_field}' НЕ найдено!")
                print(f"   Доступные поля: {list(records[0].keys())}")

        # Проверка казахских букв в текстах
        kaz_count = 0
        for rec in records:
            txt = rec.get(self.text_field, '')
            if _contains_kazakh(txt, self.min_kaz_chars):
                kaz_count += 1
        print(f"\n  Записей с казахскими буквами: {kaz_count}/{len(records)}")

        # Показать предложения из первой записи
        first_text = _normalize(records[0].get(self.text_field, ''))
        sents = _split_sentences(first_text)
        print(f"\n  Предложений из первой записи: {len(sents)}")
        for s in sents[:3]:
            print(f"    • {s[:100]}")

        print(f"\n{'─'*60}\n")

    # ── Основная загрузка ─────────────────────────────────────────────────────

    def stream_sentences(self, limit: Optional[int] = None) -> Iterator[str]:
        """Потоково отдавать отфильтрованные казахские предложения."""
        dataset = self._load_dataset()

        found = 0           # предложений найдено
        records_scanned = 0 # записей просмотрено
        rejected_lang = 0   # отброшено по языку
        rejected_len = 0    # отброшено по длине
        rejected_kaz = 0    # отброшено по отсутствию казахских букв
        field_warned = False
        t_start = time.time()
        t_last_heartbeat = t_start

        for record in dataset:
            if limit is not None and found >= limit:
                return

            records_scanned += 1

            # Heartbeat — показываем что процесс жив
            if self.heartbeat > 0 and records_scanned % self.heartbeat == 0:
                elapsed = time.time() - t_start
                speed = records_scanned / elapsed if elapsed > 0 else 0
                print(
                    f"\r  ⟳ записей: {records_scanned:,} | "
                    f"найдено: {found:,}/{limit or '∞'} | "
                    f"скорость: {speed:.0f} зап/с | "
                    f"отброшено: lang={rejected_lang} len={rejected_len} kaz={rejected_kaz}",
                    end='', flush=True
                )

            # Фильтр по языку
            if self.language_field and self.target_language:
                lang_val = record.get(self.language_field)
                if lang_val is None and not field_warned:
                    warnings.warn(
                        f"Поле '{self.language_field}' отсутствует. "
                        f"Передайте language_field=None чтобы отключить языковой фильтр."
                    )
                    field_warned = True
                if lang_val != self.target_language:
                    rejected_lang += 1
                    continue

            raw_text = record.get(self.text_field, '')
            if not raw_text:
                continue

            text = _normalize(raw_text)
            sentences = _split_sentences(text)

            for sent in sentences:
                if limit is not None and found >= limit:
                    return

                if len(sent) < self.min_sent_len:
                    rejected_len += 1
                    continue
                if len(sent) > self.max_sent_len:
                    rejected_len += 1
                    continue
                if self.min_kaz_chars > 0 and not _contains_kazakh(sent, self.min_kaz_chars):
                    rejected_kaz += 1
                    continue

                yield sent
                found += 1

        # Итоговая строка после завершения
        if self.heartbeat > 0:
            elapsed = time.time() - t_start
            print(
                f"\r  ✓ Завершено: записей={records_scanned:,} | "
                f"найдено={found:,} | "
                f"время={elapsed:.1f}с | "
                f"отброшено: lang={rejected_lang} len={rejected_len} kaz={rejected_kaz}"
                + " " * 20
            )

    def collect(self, limit: int = 10000, verbose: bool = True) -> List[str]:
        """Собрать список предложений в память."""
        sentences = []
        bar = _make_bar(limit, "Предложений найдено", verbose)

        try:
            for sent in self.stream_sentences(limit=limit):
                sentences.append(sent)
                if bar:
                    bar.update(1)
        finally:
            if bar:
                bar.close()

        if not TQDM_AVAILABLE and verbose:
            print(f"  Загружено: {len(sentences)} предложений")

        return sentences

    def save_to_file(
        self,
        output_path: str,
        limit: int = 100000,
        verbose: bool = True,
    ) -> int:
        """Сохранить предложения в текстовый файл. Возвращает количество строк."""
        count = 0
        bar = _make_bar(limit, "Предложений найдено", verbose)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for sent in self.stream_sentences(limit=limit):
                    f.write(sent + '\n')
                    count += 1
                    if bar:
                        bar.update(1)
        finally:
            if bar:
                bar.close()

        if verbose:
            print(f"  Файл: {output_path} ({count} предложений)")

        return count

    # ── Внутренние методы ─────────────────────────────────────────────────────

    def _load_dataset(self):
        load_kwargs = dict(split=self.split, streaming=True)
        if self.token:
            load_kwargs['token'] = self.token
        if self.trust_remote_code:
            load_kwargs['trust_remote_code'] = True
        return load_dataset(self.dataset_name, **load_kwargs)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bar(total: int, desc: str, verbose: bool):
    """tqdm показывает предложения найденные, не записи просмотренные."""
    if not verbose or not TQDM_AVAILABLE:
        return None
    return tqdm(
        total=total,
        desc=desc,
        unit=" пред",
        bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        dynamic_ncols=True,
        file=sys.stdout,
    )
