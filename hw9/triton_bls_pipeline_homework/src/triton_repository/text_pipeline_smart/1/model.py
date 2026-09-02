import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

class TritonPythonModel:
    def initialize(self, args):
        # Инициализируем ДВА независимых токенизатора для разных архитектур
        self.bert_tokenizer = AutoTokenizer.from_pretrained("Xenova/bert-base-multilingual-uncased-sentiment")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained("onnx-community/Qwen2.5-0.5B-Instruct")
        self.executor = ThreadPoolExecutor(max_workers=4)
        print("🔍 Продакшн-оркестратор успешно инициализирован с раздельными словарями BERT и Qwen.", flush=True)

    def execute(self, requests):
        responses = []
        
        for request in requests:
            try:
                # 1. Извлекаем входящий текстовый промпт от клиента
                prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
                if prompt_tensor is None:
                    raise ValueError("Тензор 'prompt' отсутствует в запросе.")
                    
                prompts_np = prompt_tensor.as_numpy()
                prompts = [p.decode('utf-8') if isinstance(p, bytes) else str(p) for p in prompts_np.flatten().tolist()]
                gen_text = prompts

                # 2. ПОДГОТОВКА ДАННЫХ ДЛЯ BERT (Эмбеддер + Классификатор)
                bert_tokens = self.bert_tokenizer(gen_text, return_tensors="np", truncation=True, max_length=512)
                bert_input_ids = bert_tokens["input_ids"].astype(np.int64)
                bert_attn_mask = bert_tokens["attention_mask"].astype(np.int64)
                bert_type_ids = bert_tokens["token_type_ids"].astype(np.int64)

                triton_bert_ids = pb_utils.Tensor("input_ids", bert_input_ids)
                triton_bert_mask = pb_utils.Tensor("attention_mask", bert_attn_mask)
                triton_bert_type = pb_utils.Tensor("token_type_ids", bert_type_ids)

                # 3. ПОДГОТОВКА ДАННЫХ ДЛЯ QWEN (Модель генерации)
                qwen_tokens = self.qwen_tokenizer(gen_text, return_tensors="np", truncation=True, max_length=512)
                qwen_input_ids = qwen_tokens["input_ids"].astype(np.int64)
                qwen_attn_mask = qwen_tokens["attention_mask"].astype(np.int64)

                triton_qwen_ids = pb_utils.Tensor("input_ids", qwen_input_ids)
                triton_qwen_mask = pb_utils.Tensor("attention_mask", qwen_attn_mask)

                # --- ЭТАП В: Параллельный вызов Эмбеддера и Классификатора ---
                embed_request = pb_utils.InferenceRequest(
                    model_name="text_embedder", requested_output_names=["last_hidden_state"], 
                    inputs=[triton_bert_ids, triton_bert_mask, triton_bert_type]
                )
                
                class_request = pb_utils.InferenceRequest(
                    model_name="text_classifier", requested_output_names=["logits"], 
                    inputs=[triton_bert_ids, triton_bert_mask, triton_bert_type]
                )

                def run_sync_model(req):
                    resp = req.exec()
                    if resp.has_error():
                        raise pb_utils.TritonModelException(resp.error().message())
                    return resp

                future_embed = self.executor.submit(run_sync_model, embed_request)
                future_class = self.executor.submit(run_sync_model, class_request)

                embed_response = future_embed.result()
                class_response = future_class.result()

                embeddings = pb_utils.get_output_tensor_by_name(embed_response, "last_hidden_state").as_numpy()
                logits = pb_utils.get_output_tensor_by_name(class_response, "logits").as_numpy()

                # --- ЭТАП Г: Вызов третьей модели - Генератора текста (Qwen2.5-ONNX) ---
                generator_request = pb_utils.InferenceRequest(
                    model_name="text_generator", 
                    requested_output_names=["logits"], 
                    inputs=[triton_qwen_ids, triton_qwen_mask] # Передаем РОДНЫЕ токены Qwen!
                )
                
                generator_response = run_sync_model(generator_request)
                gen_logits = pb_utils.get_output_tensor_by_name(generator_response, "logits").as_numpy()

                # Формируем статус-ответ для клиента
                success_msg = f"Инференс LLM Qwen2.5 выполнен успешно! Матрица логитов токенов генератора: {gen_logits.shape}"
                clean_text_array = np.array([success_msg.encode('utf-8')], dtype=object)

                # --- Сборка финального ответа ---
                output_tensor_text = pb_utils.Tensor("generated_text", clean_text_array)
                output_tensor_embed = pb_utils.Tensor("embeddings", embeddings)
                output_tensor_logits = pb_utils.Tensor("logits", logits)

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor_text, output_tensor_embed, output_tensor_logits]
                )
                responses.append(inference_response)

            except Exception as e:
                err_msg = str(e)
                print(f"💥 [Pipeline Error]: {err_msg}", flush=True)
                triton_error = pb_utils.TritonError(f"[Pipeline Error] -> {err_msg}")
                responses.append(pb_utils.InferenceResponse(output_tensors=[], error=triton_error))
                
        return responses

    def finalize(self):
        self.executor.shutdown(wait=True)
        print("Cleaning up text_pipeline_smart.")
