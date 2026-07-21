import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer  # Supervised Fine-Tuning Trainer

# Добавление Fine-Tuning в проект
# src/infrastructure/llm/local_client.py

#from peft import PeftModel

# Внутри __init__ вашего LocalLLMClient:
#base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

# Минимальное изменение: склеиваем базовую модель и ваш обученный LoRA адаптер стиля
#adapter_path = "/mnt/d/ai_models/qwen_corporate_adapter"
#if os.path.exists(adapter_path):
#    print("🎭 Подключение обученного адаптера тона и структуры ответов...")
#    self.model = PeftModel.from_pretrained(base_model, adapter_path).to(self.device)
#else:
#    self.model = base_model.to(self.device)

def run_finetuning():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    cache_dir = os.getenv("HF_HOME", "./storage/models")
    output_dir = "/mnt/d/ai_models/qwen_corporate_adapter"

    print("📥 Загрузка датасета идеальных корпоративных ответов...")
    # Датасет должен содержать колонку "text" с правильными диалогами:
    # <система:будь вежлив> <пользователь:вопрос> <ассистент:идеальный ответ [Док.pdf, стр.1]>
    dataset = load_dataset("json", data_files="data/finetune_dataset.jsonl", split="train")

    print("🤖 Загрузка токенизатора и модели в 4-битном режиме для экономии VRAM...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # Конфигурация квантования, чтобы модель влезла в видеокарту ноутбука
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        load_in_4bit=True, # Включаем 4-битное сжатие
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Подготовка модели к обучению адаптеров
    model = prepare_model_for_kbit_training(model)

    print("📐 Настройка LoRA (обучаемой матрицы весов)...")
    # Указываем слои модели Qwen, в которые внедрим новые поведенческие привычки
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Слои внимания (Attention)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    print("⚙️ Конфигурация параметров обучения...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, # Минимальный батч для экономии памяти ноутбука
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,                  # Количество шагов обучения
        learning_rate=2e-4,
        logging_steps=10,
        fp16=True,
        save_strategy="no"
    )

    print("🚀 Старт обучения характера модели...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()

    # Сохраняем только обученный LoRA-адаптер (он весит всего 10-20 МБ)
    trainer.model.save_pretrained(output_dir)
    print(f"✅ Обучение завершено! Адаптер стиля сохранен в: {output_dir}")

if __name__ == "__main__":
    run_finetuning()


