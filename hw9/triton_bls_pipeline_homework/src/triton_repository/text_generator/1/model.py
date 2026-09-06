import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import pipeline

class TritonPythonModel:
    def initialize(self, args):
        # Загружаем реальную, живую малую LLM Qwen на CPU
        self.generator = pipeline(
            "text-generation", 
            model="Qwen/Qwen2.5-0.5B-Instruct", 
            device="cpu"
        )
        print("🚀 Реальная модель Qwen-0.5B успешно загружена в RAM сервера!", flush=True)

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                # Извлекаем входящий промпт
                prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
                prompt_str = prompt_tensor.as_numpy()[0].decode('utf-8')

                # Честный инференс реальной LLM Qwen
                outputs = self.generator(prompt_str, max_new_tokens=32, do_sample=True, temperature=0.7)
                generated_text = outputs[0]["generated_text"]

                # Упаковываем ответ
                out_tensor = pb_utils.Tensor("generated_text", np.array([generated_text.encode('utf-8')], dtype=object))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
                
            except Exception as e:
                triton_error = pb_utils.TritonError(f"Ошибка генератора Qwen: {str(e)}")
                responses.append(pb_utils.InferenceResponse(output_tensors=[], error=triton_error))
                
        return responses

    def finalize(self):
        print("Cleaning up text_generator.")


#import triton_python_backend_utils as pb_utils

#class TritonPythonModel:
#    # Делаем метод статическим, как требует C++ ядро Triton 25.12
#    @staticmethod
#    def auto_complete_config(config):
#        input_meta = {"name": "prompt", "data_type": "TYPE_STRING", "dims": [-1]}
#        output_meta = {"name": "generated_text", "data_type": "TYPE_STRING", "dims": [-1]}
#        
#        config.set_max_batch_size(0)
#        config.add_input(input_meta)
#        config.add_output(output_meta)
#        return config
#
#    def execute(self, requests):
#        responses = []
#        for request in requests:
#            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
#            out_tensor = pb_utils.Tensor("generated_text", prompt_tensor.as_numpy())
#            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
#        return responses
