from openai import OpenAI
import time
import random
import threading
# vllm serve model/llama-2-13b-chat-hf --served-model-name llama-2-13b-chat
# vllm serve model/llama-2-13b-chat-hf --served-model-name llama-2-13b-chat

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
model = "llama-2-13b-chat"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def generate(prompt, model_name="gpt-4o-mini", timeout=100, retry=10):
    for i in range(retry):
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
                ],
            max_tokens=1000,
            temperature=0,
            top_p=0.95,
            extra_body={
                "top_k": 20,
                },
            stop=['\n}'],
            timeout=timeout,
            )

        if not completion.choices or len(completion.choices) == 0:
            continue
        else:
            text = completion.choices[0].message.content
            return text
    print("No reply from GPT")
    return """{
        "Lable": "None",
        "Thought": "None",
        "Hallucination": False,
        }"""

if __name__ == "__main__":
    prompt = "Explain the theory of relativity in simple terms."
    response = generate(prompt)
    print(response)