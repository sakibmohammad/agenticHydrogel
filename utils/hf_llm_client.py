"""
hf_llm_client.py

Simple Hugging Face LLM client for inference.
"""

import os
import requests

class HFLLMClient:
    def __init__(self, model_id=None, hf_token=None):
        self.model_id = model_id or os.getenv(
            "HF_MODEL_ID",
            "mistralai/Mistral-7B-Instruct-v0.3"
        )
        self.hf_token = hf_token or os.getnv("HF_TOKEN")

        if self.hf_token is None:
            raise ValueError("Hugging Face token not available. Please set HF token before using the client.")
        
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }

        def generate(self, prompt, max_new_token=512, temperature=0.7):
            payload = {
                "inputs": prompt,
                "parameters":{
                    "max_new_tokens": max_new_token,
                    "temperature": temperature,
                    "return_full_text": False
                }
            }

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=120
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"HF API error {response.status_code}: {response.text}"
                )
            
            result = response.json()

            if isinstance(result, list) and "generated_text" in result[0]:
                result = result[0]["generated_text"]

            if isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            
            return str(result)
        
def test_hf_client():
    client = HFLLMClient()
    prompt = "Explain the concept of multi-agent-based system in a single sentence."
    output = client.generate(prompt)
    print(f"Prompt: {prompt}\nOutput: {output}")

if __name__ == "__main__":
    test_hf_client()