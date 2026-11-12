import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from torch.optim import AdamW

# =========================
# CONFIG (ОПТИМИЗИРОВАНО ПОД KAGGLE T4/P100 — 16GB VRAM)
# =========================
MODEL_NAME = "google/mt5-small"
BATCH_SIZE = 4          # fp32 + mt5-small: безопасный размер для T4
GRAD_ACCUM_STEPS = 8    # Эффективный батч = 4 * 8 = 32
EPOCHS = 5
MAX_LEN = 128
LR = 5e-5
NUM_WORKERS = 2         # 2 лучше чем 4 в Kaggle notebooks
SAVE_DIR = "gec_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Устройство: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =========================
# DATASET
# =========================
class GECDataset(Dataset):
    def __init__(self, path, tokenizer, max_samples=None):
        raw_data = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if line.strip():
                    raw_data.append(json.loads(line))
        print(f"Загружено примеров: {len(raw_data)}, токенизация...")

        corrupted_texts = [f"gec: {d['corrupted']}" for d in raw_data]
        original_texts = [d["original"] for d in raw_data]

        enc_inputs = tokenizer(
            corrupted_texts,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
        )
        enc_targets = tokenizer(
            original_texts,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
        )

        self.input_ids = torch.tensor(enc_inputs["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(enc_inputs["attention_mask"], dtype=torch.long)
        labels = torch.tensor(enc_targets["input_ids"], dtype=torch.long)
        labels[labels == tokenizer.pad_token_id] = -100
        self.labels = labels
        print("Токенизация завершена.")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

# =========================
# TRAIN
# =========================
def train(data_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    dataset = GECDataset(data_path, tokenizer)

    # Разбивка на train/val (90/10)
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Исключаем bias и LayerNorm из weight_decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)

    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    print("Начинаем обучение...")
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{EPOCHS} [train]")

        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            progress.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.4f}")

        # Обрабатываем остаток батчей
        if (step + 1) % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        avg_train_loss = total_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Эпоха {epoch+1}/{EPOCHS} [val]"):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        torch.cuda.empty_cache()
        print(f"Эпоха {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # Сохраняем лучшую модель
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"  Модель сохранена (val_loss={best_val_loss:.4f})")

    print(f"Готово! Лучшая модель сохранена в {SAVE_DIR}/")

if __name__ == "__main__":
    train("data/corpus.jsonl")
