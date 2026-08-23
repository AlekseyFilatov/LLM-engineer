import json
import os
import matplotlib.pyplot as plt

state_path = './src/storage/ollama_models/lora_output/checkpoint-69/trainer_state.json'
if not os.path.exists(state_path):
    state_path = './src/storage/lora_output/checkpoint-69/trainer_state.json'

try:
    with open(state_path, 'r', encoding='utf-8') as f:
        state_data = json.load(f)
    
    epochs = [log['epoch'] for log in state_data['log_history'] if 'loss' in log]
    losses = [log['loss'] for log in state_data['log_history'] if 'loss' in log]
    
    print(f'📊 [LLMOPS] Найдено точек в истории trainer_state: {len(losses)} из 69')
    
    plt.figure(figsize=(11, 5))
    plt.plot(epochs, losses, color='#2ca02c', linewidth=2, label='Train Loss (Шаги 1-69)')
    plt.scatter(epochs[-1], losses[-1], color='red', s=50, zorder=5, label=f'Финал 3 эпохи (Loss: {losses[-1]:.5f})')
    
    plt.title('E-Corp ECQL SFT Fine-Tuning: Полный график сходимости (Эпохи 1-3)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Эпохи обучения (Epochs)', fontsize=11)
    plt.ylabel('Значение функции потерь (Loss)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    
    plt.savefig('loss_curve_full.png', dpi=300, bbox_inches='tight')
    print('✨ [SUCCESS] Сквозной график за все 3 эпохи успешно сохранен в файл: loss_curve_full.png')

except Exception as e:
    print(f'⚠️ Не удалось прочитать trainer_state.json ({e}). Строим по точкам 3-й эпохи...')
    losses_3 = [0.001125, 0.02026, 0.002538, 0.0282, 0.0117, 0.001098, 0.01123, 0.0008726, 0.002698]
    epochs_3 = [2.652, 2.696, 2.739, 2.783, 2.826, 2.87, 2.913, 2.957, 3.0]
    
    plt.figure(figsize=(10, 4))
    plt.plot(epochs_3, losses_3, marker='o', color='#17becf', linewidth=2, label='Train Loss (3-я Эпоха)')
    plt.title('Динамика сходимости на 3-й эпохе обучения (Выход на плато)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Эпохи обучения (Epochs)', fontsize=11)
    plt.ylabel('Значение функции потерь (Loss)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    
    plt.savefig('loss_curve_full.png', dpi=300, bbox_inches='tight')
    print('✨ [SUCCESS] График 3-й эпохи успешно запечен в файл: loss_curve_full.png')
