import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration

# 1. Путь к модели
model_path = r"C:\Users\Arlan Alimbay\Documents\kaz-error-generator\gec_model_final" 

# 2. Загрузка
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    model.eval()
    
    print(f"✅ Модель успешно загружена на: {DEVICE}")
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")

def fix_errors(text):
    # Убираем префикс "gec: ", так как его нет в датасете
    input_text = text 

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=5,           # Увеличиваем поиск
        temperature=0.9,        # Добавляем немного смелости
        do_sample=True,         # Разрешаем выбирать не только самый вероятный вариант
        top_p=0.95,
        repetition_penalty=1.0  # Убираем штраф совсем
)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 3. Тестирование
test_sentences = [
    "Бір аншы болыпты",           # Ожидаем: Сенің анаң жұмыста.
    "Өмир бойы аң аулап, күн көріпті.",    # Ожидаем: Балалар кітаптарды оқып отыр.
    "Қандай дәле керек саған? – дейді арыстан мақтанып." # Ожидаем: Қазақстанның елордасы — Астана қаласы.
]

print("\n" + "="*30)
print("   РЕЗУЛЬТАТЫ ПРОВЕРКИ GEC")
print("="*30)

for sent in test_sentences:
    fixed = fix_errors(sent)
    print(f"ВХОД:  {sent}")
    print(f"ВЫХОД: {fixed}")
    
    # Небольшая проверка: изменилось ли что-то
    if sent == fixed:
        print("⚠️ Статус: Без изменений (Identity mapping)")
    else:
        print("✨ Статус: Исправлено")
    print("-" * 30)