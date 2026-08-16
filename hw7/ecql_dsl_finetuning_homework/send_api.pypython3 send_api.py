import json
import requests

def run():
    url = 'http://127.0.0.1:11434/api/create'
    payload = {
        'name': 'e-corp-coder',
        'modelfile': 'FROM qwen2.5-coder:7b\nADAPTER /home/alexfil/LLM-Training/src/storage/ollama_models/lora_output\nPARAMETER temperature 0.1\nSYSTEM "Ты — официальный ИИ-транслятор компании E-Corp. Твоя единственная задача — строго переводить запросы пользователя на язык ECQL. Выдавай СТРОГО чистый код ECQL."'
    }
    
    print('📡 [API] Отправка чистого JSON-пакета в Ollama...')
    try:
        response = requests.post(url, json=payload, stream=True, timeout=120)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    status = json.loads(line.decode('utf-8'))
                    if "status" in status:
                        print(f"   [OLLAMA]: {status['status']}")
        else:
            print(f"❌ Ошибка сервера Ollama ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"❌ Критический сбой: {e}")

if __name__ == "__main__":
    run()

