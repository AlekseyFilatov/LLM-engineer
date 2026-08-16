# Создаем целевую директорию
mkdir -p ~/LLM-Training/src/storage/lora_output

# Перемещаем сохраненные веса туда, где их ждет конвертер
mv ~/LLM-Training/storage/ecql_lora_output/* ~/LLM-Training/src/storage/lora_output/

# Удаляем пустую старую папку из корня
rm -rf ~/LLM-Training/storage/
