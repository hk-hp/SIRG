# CUDA_VISIBLE_DEVICES=3 vllm serve /model/Llama-3.1-8B-Instruct  --served-model-name Llama-3.1-8B
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def get_answer(mess, temperature=0.0):
    chat_response = client.chat.completions.create(
    model="Llama-3.1-8B",
    messages=[
        # {"role": "system", "content": sys_prompt},
        {"role": "user", "content": mess},
    ],
    max_tokens=3000,
    temperature=temperature,
    top_p=0.95,
    extra_body={
        "top_k": 20,
    },
    )
    return chat_response.choices[0].message.content