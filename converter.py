import json
import random
from pathlib import Path
import difflib


def tokenize(text: str):
    """Токенизация (простая, word-level)"""
    return text.strip().split()


def align_and_label(corrupted_tokens, original_tokens):
    """
    Выравнивание через difflib + генерация labels
    """

    matcher = difflib.SequenceMatcher(
        None, corrupted_tokens, original_tokens
    )

    tokens_out = []
    labels_out = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        # KEEP
        if tag == "equal":
            for i in range(i1, i2):
                tokens_out.append(corrupted_tokens[i])
                labels_out.append("KEEP")

        # REPLACE
        elif tag == "replace":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                tokens_out.append(corrupted_tokens[i])
                labels_out.append(f"REPLACE_{original_tokens[j]}")

        # DELETE (лишние токены в corrupted)
        elif tag == "delete":
            for i in range(i1, i2):
                tokens_out.append(corrupted_tokens[i])
                labels_out.append("DELETE")

        # INSERT (не хватает токенов → вставка)
        elif tag == "insert":
            for j in range(j1, j2):
                # вставка прикрепляется к предыдущему токену
                if tokens_out:
                    labels_out[-1] += f"|APPEND_{original_tokens[j]}"
                else:
                    # редкий случай: вставка в начало
                    tokens_out.append(original_tokens[j])
                    labels_out.append(f"APPEND_{original_tokens[j]}")

    return tokens_out, labels_out


def convert_record(record):
    corrupted = record["corrupted"]
    original = record["original"]

    corrupted_tokens = tokenize(corrupted)
    original_tokens = tokenize(original)

    # ❗ фильтр слишком длинных
    if len(corrupted_tokens) > 100 or len(original_tokens) > 100:
        return None

    tokens, labels = align_and_label(corrupted_tokens, original_tokens)

    if not tokens or not labels:
        return None

    return " ".join(tokens), " ".join(labels)


def convert_file(
    input_path,
    train_path="train.txt",
    dev_path="dev.txt",
    split_ratio=0.9,
    max_samples=None,
):
    data = []

    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            record = json.loads(line)

            converted = convert_record(record)
            if converted:
                data.append(converted)

            if i % 1000 == 0 and i > 0:
                print(f"Обработано: {i}")

    print(f"✅ Готово примеров: {len(data)}")

    random.shuffle(data)

    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    dev_data = data[split_idx:]

    def save(path, dataset):
        with open(path, "w", encoding="utf-8") as f:
            for tokens, labels in dataset:
                f.write(tokens + "\n")
                f.write(labels + "\n\n")

    save(train_path, train_data)
    save(dev_path, dev_data)

    print(f"📁 Train: {len(train_data)}")
    print(f"📁 Dev: {len(dev_data)}")


if __name__ == "__main__":
    convert_file(
        input_path="data/corpus.jsonl",
        train_path="data/train.txt",
        dev_path="data/dev.txt",
        split_ratio=0.9,
    )