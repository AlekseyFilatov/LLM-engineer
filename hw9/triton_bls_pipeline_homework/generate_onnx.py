import torch
import torch.nn as nn
import os

# 1. Задаем пути к вашим папкам моделей
BASE_DIR = os.path.expanduser("~/LLM-Training/src/triton_repository")
embedder_path = os.path.join(BASE_DIR, "text_embedder/1/model.onnx")
classifier_path = os.path.join(BASE_DIR, "text_classifier/1/model.onnx")

# Убедимся, что папки существуют
os.makedirs(os.path.dirname(embedder_path), exist_ok=True)
os.makedirs(os.path.dirname(classifier_path), exist_ok=True)

# Опишем фейковые входные тензоры (батч=1, длина последовательности=4)
dummy_input_ids = torch.ones(1, 4, dtype=torch.int64)
dummy_attn_mask = torch.ones(1, 4, dtype=torch.int64)

# --- Создаем граф для Эмбеддера ---
class MockEmbedder(nn.Module):
    def forward(self, input_ids, attention_mask):
        # Возвращаем фиксированный размер [-1, 384] на основе длины последовательности
        seq_len = input_ids.size(1)
        return torch.zeros(1, seq_len, 384, dtype=torch.float32)

# --- Создаем граф для Классификатора ---
class MockClassifier(nn.Module):
    def forward(self, input_ids, attention_mask):
        # Возвращаем фиксированный размер [6] для логитов классов
        return torch.zeros(1, 6, dtype=torch.float32)

# Экспортируем Эмбеддер в ONNX
torch.onnx.export(
    MockEmbedder(), (dummy_input_ids, dummy_attn_mask), embedder_path,
    input_names=['input_ids', 'attention_mask'], output_names=['last_hidden_state'],
    dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}, 'attention_mask': {0: 'batch', 1: 'sequence'}, 'last_hidden_state': {0: 'batch', 1: 'sequence'}}
)

# Экспортируем Классификатор в ONNX
torch.onnx.export(
    MockClassifier(), (dummy_input_ids, dummy_attn_mask), classifier_path,
    input_names=['input_ids', 'attention_mask'], output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}, 'attention_mask': {0: 'batch', 1: 'sequence'}, 'logits': {0: 'batch'}}
)

print("✅ Валидные ONNX-модели успешно сгенерированы и записаны поверх пустых файлов!")
