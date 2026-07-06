"""
ollama_llm_client.py

Local LLM client for hydrogel material workflow.
"""

import json
import urllib.request
import urllib.error

class OllamaLLMClient:
    """
    Ollama client using Python standart library.
    """

    def __init__(
        self,
        model_name="qwen2.5:1.5b",
        base_url="http://localhost:11434",
        timeout=120
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self):
        """
        Check if Ollama is running locally.
        """    

        try:
            url = f"{self.base_url}/api/tags"
            request = urllib.request.Request(url, method="GET")

            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status == 200
        
        except Exception:
            return False


    def generate(self, prompt, system_prompt=None, temperature=0.2, max_tokens=512):
        """
        Generate a response from Ollama.
        """    

        full_prompt = prompt

        if system_prompt is not None:
            full_prompt = f"{system_prompt}\n\nUser request:\n{prompt}"
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }     

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url=f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
                )
        
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))

            return response_data.get("response", "").strip()

        except urllib.error.URLError as error:
            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure Ollama is installed and running."
            ) from error

    def safe_generate(self,
                      prompt,
                      system_prompt=None,
                      temperature=0.2,
                      max_tokens=512,
                      fallback_text=None):
        """
        Generate text if Ollama is available else return fallback_text instead of crashing.
        """    

        if not self.is_available():
            if fallback_text is not None:
                return fallback_text
            
            return (
                "Ollama is not available. The workflow still completed because "
                "the LLM is optional."
            )
        
        try:
            return self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        except Exception as error:
            if fallback_text is not None:
                return fallback_text
            return f"LLM summary is not available: {error}"
        
def summarize_workflow_with_ollama(
    workflow_summary,
    model_name="qwen2.5:1.5b"
):
    """
    Summarize a completed workflow using Ollama.
    """

    client = OllamaLLMClient(model_name=model_name)
            
    system_prompt = """
    You are a concise scientific workflow assistant.
    Summarize the hydrogel machine learning workflow in clear language.
    Do not invent results.
    Only use the information provided.
    """

    prompt = f"""Summarize this workflow result for a researcher: {json.dumps(workflow_summary, indent=2)}"""

    return client.safe_generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=512,
        fallback_text="LLM summary skipped because Ollama is not available."
    )

if __name__ == "__main__":
    client = OllamaLLMClient()

    print(f"Ollama available: {client.is_available()}")

    response = client.safe_generate(
        prompt="Explain what a conditional VAE does in one short paragraph.",
        fallback_text="Ollama is not running, but the client script works."
    )

    print(response)       