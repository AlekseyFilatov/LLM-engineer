#!/bin/bash
echo "🚨 ВНИМАНИЕ: Вы запускаете полную очистку проекта!"
read -p "Вы уверены, что хотите удалить базу данных, кэш моделей и окружение? (y/n): " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo "🗑️ Удаление локальной векторной базы ChromaDB и логов..."
    rm -rf storage/
    
    echo "🗑️ Удаление виртуального окружения Python (ml_env)..."
    rm -rf ml_env/
    
    # Читаем путь к кэшу моделей из .env или удаляем стандартный на диске D
    if [ -d "/mnt/d/ai_models/huggingface" ]; then
        echo "🗑️ Удаление скачанных весов LLM моделей с диска D (Папка ai_models)..."
        rm -rf /mnt/d/ai_models/huggingface
    fi
    
    echo "🗑️ Удаление кэша Python (__pycache__)..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    
    echo "✨ Проект полностью очищен! База, веса моделей и окружение удалены."
    echo "ℹ️ Теперь вы можете заново инициализировать чистый проект."
else
    echo "❌ Очистка отменена."
fi
