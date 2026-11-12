"""
KazakhErrorGenerator — синтетический генератор ошибок для казахского текста.

Применяет правила из error_rules.py к токенам входного текста и возвращает
объект GenerationResult с исходным текстом, искажённым текстом и аннотациями.
"""

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .error_rules import (
    KAZ_SPECIAL_CHAR_MAP,
    TYPO_TYPES,
    DEFAULT_ERROR_TYPE_WEIGHTS,
    GRAMMAR_RULES,
)


# ── Типы данных ───────────────────────────────────────────────────────────────

@dataclass
class ErrorAnnotation:
    """Аннотация одной ошибки."""
    token_index: int       # индекс токена в тексте
    char_offset: int       # позиция в символах от начала текста
    original: str          # исходная форма
    corrupted: str         # искажённая форма
    error_type: str        # категория: special_chars, typos, ...
    error_subtype: str     # подтип: drop_comma, double, ң→н, ...


@dataclass
class GenerationResult:
    """Результат генерации ошибок для одного текста."""
    original: str
    corrupted: str
    annotations: List[ErrorAnnotation] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.annotations)

    @property
    def error_rate(self) -> float:
        tokens = self.original.split()
        return self.error_count / len(tokens) if tokens else 0.0

    def to_dict(self) -> dict:
        return {
            'original': self.original,
            'corrupted': self.corrupted,
            'error_count': self.error_count,
            'annotations': [
                {
                    'token_index': a.token_index,
                    'char_offset': a.char_offset,
                    'original': a.original,
                    'corrupted': a.corrupted,
                    'error_type': a.error_type,
                    'error_subtype': a.error_subtype,
                }
                for a in self.annotations
            ]
        }


# ── Вспомогательные функции ───────────────────────────────────────────────────

# Казахский алфавит (кириллица) для случайных вставок
KAZ_ALPHABET = 'аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя'

_KAZAKH_WORD_RE = re.compile(
    r'[а-яёА-ЯЁәіңғқөүұәІҢҒҚӨҮҰ]+',
    re.UNICODE
)

_PUNCT_RE = re.compile(r'[,;:!?]')
_SENTENCE_END_RE = re.compile(r'\.\s')


def _is_kazakh_word(token: str) -> bool:
    """Содержит ли токен хотя бы одну казахскую/кириллическую букву."""
    return bool(_KAZAKH_WORD_RE.search(token))


def _weighted_choice(weights: Dict[str, float]) -> str:
    keys = list(weights.keys())
    vals = list(weights.values())
    return random.choices(keys, weights=vals, k=1)[0]


# ── Основной класс ────────────────────────────────────────────────────────────

class KazakhErrorGenerator:
    """
    Синтетический генератор ошибок для казахского текста.

    Параметры:
        token_error_rate  — вероятность того, что конкретный токен получит ошибку
        error_type_weights — относительные веса типов ошибок
        seed              — зерно генератора случайных чисел (для воспроизводимости)

    Использование:
        gen = KazakhErrorGenerator(token_error_rate=0.15)
        result = gen.generate("Бүгін ауа райы өте жақсы.")
        print(result.corrupted)
        print(result.annotations)
    """

    def __init__(
        self,
        token_error_rate: float = 0.15,
        error_type_weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ):
        self.token_error_rate = token_error_rate
        self.error_type_weights = error_type_weights or DEFAULT_ERROR_TYPE_WEIGHTS.copy()
        if seed is not None:
            random.seed(seed)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, text: str) -> GenerationResult:
        """Сгенерировать ошибки для одного текста."""
        # Токенизация с сохранением позиций
        tokens, offsets = self._tokenize(text)
        corrupted_tokens = list(tokens)
        annotations: List[ErrorAnnotation] = []

        for i, token in enumerate(tokens):
            if not _is_kazakh_word(token):
                # Пунктуационные/пробельные ошибки обрабатываются отдельно
                continue

            if random.random() > self.token_error_rate:
                continue

            error_type = _weighted_choice(self.error_type_weights)
            result = self._apply_error(token, error_type)

            # Для grammar — не делаем fallback: если правило не подошло,
            # пропускаем токен. Иначе grammar вытесняется typos/special_chars.
            if result is None and error_type != 'grammar':
                available_types = [t for t in self.error_type_weights.keys()
                                   if t != error_type and t != 'grammar']
                random.shuffle(available_types)
                for fallback_type in available_types:
                    result = self._apply_error(token, fallback_type)
                    if result is not None:
                        error_type = fallback_type
                        break

            if result is None:
                continue

            new_token, subtype = result
            if new_token == token:
                continue

            corrupted_tokens[i] = new_token
            annotations.append(ErrorAnnotation(
                token_index=i,
                char_offset=offsets[i],
                original=token,
                corrupted=new_token,
                error_type=error_type,
                error_subtype=subtype,
            ))

        # Пунктуационные ошибки применяем к всему тексту отдельно
        corrupted_text = self._rejoin(corrupted_tokens, text, tokens, offsets)
        corrupted_text, punct_annotations = self._apply_punctuation_errors(
            corrupted_text, len(annotations)
        )
        annotations.extend(punct_annotations)

        return GenerationResult(
            original=text,
            corrupted=corrupted_text,
            annotations=annotations,
        )

    def generate_batch(self, texts: List[str]) -> List[GenerationResult]:
        """Сгенерировать ошибки для списка текстов."""
        return [self.generate(t) for t in texts]

    # ── Токенизация ───────────────────────────────────────────────────────────

    def _tokenize(self, text: str):
        """Разбить текст на токены с сохранением смещений."""
        tokens = []
        offsets = []
        # Разбиваем по словам и знакам, сохраняем пробелы как отдельные токены
        for m in re.finditer(r'\S+|\s+', text):
            tokens.append(m.group())
            offsets.append(m.start())
        return tokens, offsets

    def _rejoin(self, corrupted_tokens, original_text, original_tokens, offsets) -> str:
        """Собрать токены обратно в текст."""
        result = []
        pos = 0
        for i, (token, offset) in enumerate(zip(original_tokens, offsets)):
            # Добавляем текст между предыдущей и текущей позицией (пробелы уже в токенах)
            result.append(corrupted_tokens[i])
            pos = offset + len(token)
        return ''.join(result)

    # ── Применение ошибок ─────────────────────────────────────────────────────
# ── grammar: ошибки в окончаниях ──────────────────────────────────────────

    

    # Не забудь обновить метод _apply_error, чтобы он вызывал новую функцию:
    
        # ... остальные условия ...
    def _apply_error(self, token: str, error_type: str):
        """Применить ошибку указанного типа к токену. Возвращает (новый_токен, подтип) или None."""
        if error_type == 'grammar':
            return self._error_grammar(token)
        elif error_type == 'special_chars':
            return self._error_special_chars(token)
        elif error_type == 'typos':
            return self._error_typo(token)
        return None

    # ── special_chars: ң→н, ү→у и т.д. ──────────────────────────────────────

    def _error_special_chars(self, token: str):
        """Заменить одну казахскую специальную букву на русский аналог."""
        # Найти все позиции казахских букв в токене
        positions = [
            (i, ch) for i, ch in enumerate(token)
            if ch in KAZ_SPECIAL_CHAR_MAP
        ]
        if not positions:
            return None

        pos, char = random.choice(positions)
        replacement = KAZ_SPECIAL_CHAR_MAP[char]
        new_token = token[:pos] + replacement + token[pos + 1:]
        return new_token, f'{char}→{replacement}'

    # ── typos: удвоение, перестановка, пропуск, вставка ───────────────────────

    def _error_typo(self, token: str):
        """Применить случайную опечатку."""
        if len(token) < 2:
            return None

        subtype = random.choice(TYPO_TYPES)

        if subtype == 'double':
            # Удвоить случайную букву: кітап → кіттап
            pos = random.randint(0, len(token) - 1)
            new_token = token[:pos] + token[pos] + token[pos:]
            return new_token, 'double'

        elif subtype == 'transpose':
            # Поменять местами соседние буквы: кітап → ктіап
            if len(token) < 2:
                return None
            pos = random.randint(0, len(token) - 2)
            chars = list(token)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return ''.join(chars), 'transpose'

        elif subtype == 'drop':
            # Пропустить букву: кітап → ктап
            if len(token) < 3:
                return None
            pos = random.randint(0, len(token) - 1)
            new_token = token[:pos] + token[pos + 1:]
            return new_token, 'drop'

        elif subtype == 'insert':
            # Вставить случайную букву: кітап → кітааап
            pos = random.randint(0, len(token))
            char = random.choice(KAZ_ALPHABET)
            new_token = token[:pos] + char + token[pos:]
            return new_token, 'insert'

        return None

    # ── punctuation: ошибки в знаках препинания ───────────────────────────────

    def _error_grammar(self, token: str):
        """Случайная замена или удаление грамматических аффиксов."""
        from .error_rules import GRAMMAR_RULES
        
        subtype = random.choice(list(GRAMMAR_RULES.keys()))
        rules = GRAMMAR_RULES[subtype]
        
        for original_affix, corrupted_affix in rules:
            if token.endswith(original_affix):
                # Заменяем окончание
                new_token = token[:-len(original_affix)] + corrupted_affix
                return new_token, f'grammar_{subtype}:{original_affix}→{corrupted_affix}'
        
        return None
    
    def _apply_punctuation_errors(self, text: str, annotation_offset: int):
        """Применить пунктуационные ошибки ко всему тексту."""
        annotations = []
        result = text

        # drop_comma: убрать случайную запятую (вероятность 20%)
        commas = [m.start() for m in re.finditer(r',', result)]
        for pos in commas:
            if random.random() < 0.20:
                result = result[:pos] + result[pos + 1:]
                annotations.append(ErrorAnnotation(
                    token_index=-1,
                    char_offset=pos,
                    original=',',
                    corrupted='',
                    error_type='punctuation',
                    error_subtype='drop_comma',
                ))
                break  # одна ошибка за раз

        # extra_space: добавить пробел перед знаком препинания (вероятность 4%)
        for m in re.finditer(r'([^\s])([,;:!?])', result):
            if random.random() < 0.04:
                pos = m.start(2)
                result = result[:pos] + ' ' + result[pos:]
                annotations.append(ErrorAnnotation(
                    token_index=-1,
                    char_offset=pos,
                    original=m.group(2),
                    corrupted=' ' + m.group(2),
                    error_type='punctuation',
                    error_subtype='extra_space',
                ))
                break

        return result, annotations
