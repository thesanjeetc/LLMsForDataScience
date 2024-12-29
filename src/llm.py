from dotenv import load_dotenv
from io import BytesIO
from enum import Enum
from tenacity import retry, wait_exponential, stop_after_attempt, before_sleep_log
import openai
import logging
import os
import json
import instructor

logger = logging.getLogger(__name__)

load_dotenv()

openai_client = openai.OpenAI(
    api_key = os.getenv("OPENAI_KEY") 
)

groq_client = openai.OpenAI(
    base_url = "https://api.groq.com/openai/v1",
    api_key = os.getenv("GROQ_KEY")
)

anyscale_client = openai.OpenAI(
    base_url = "https://api.endpoints.anyscale.com/v1",
    api_key = os.getenv("ANYSCALE_KEY")
)

deepseek_client = openai.OpenAI(
    base_url = "https://api.deepseek.com/v1",
    api_key = os.getenv("DEEPSEEK_KEY")
)

together_client = openai.OpenAI(
  base_url='https://api.together.xyz/v1',
  api_key=os.getenv("TOGETHER_KEY")
)

deepinfra_client = openai.OpenAI(
  base_url='https://api.deepinfra.com/v1/openai',
  api_key=os.getenv("DEEPINFRA_KEY")
)

print(os.getenv("DEEPINFRA_KEY"))

clients = {
    "gpt-3.5-turbo": openai_client,
    "gpt-4-turbo": openai_client,
    "llama3-8b-8192": groq_client,
    "llama3-70b-8192": groq_client,
    "meta-llama/Llama-3-8b-chat-hf": together_client,
    "meta-llama/Llama-3-70b-chat-hf": together_client,
    "deepseek-ai/deepseek-coder-33b-instruct": together_client,
    "deepseek-coder": deepseek_client,
    "meta-llama/Meta-Llama-3-8B-Instruct": deepinfra_client,
    "meta-llama/Meta-Llama-3-70B-Instruct": deepinfra_client,
}

LLMs = {
    "LLAMA3_INSTRUCT_8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "LLAMA3_INSTRUCT_70B": "meta-llama/Meta-Llama-3-70B-Instruct",
    "DEEPSEEK_CODER_33B": "deepseek-coder"
}

SYSTEM_PROMPT = """
You are a coding assistant for data science tasks working on notebooks.
Your task is to only output Python code.
Do not output any other text other than Python comments.
"""

@retry(
    wait=wait_exponential(min=2, max=60), 
    # before_sleep=before_sleep_log(logger, logging.INFO),
    stop=stop_after_attempt(10)
)
def generate_response(model, prompt, temperature=0.6, max_tokens=1024, response_model=None):
    model_str = LLMs[model]
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [{"role": "user", "content": prompt}]
    if response_model:
        client = instructor.from_openai(clients[model_str], mode=instructor.Mode.JSON)
        return client.chat.completions.create(
            model=model_str,
            response_model=response_model,
            response_format={"type":"json_object"},
            tool_choice="auto",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=5
        )
    else:
        chat_completion = clients[model_str].chat.completions.create(
            model=model_str,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return chat_completion.choices[0].message.content

def submit_batch(reqs, uids, config):
    batch_reqs = [json.dumps({
            "custom_id": uids[i], 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": config["model"], 
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": config["max_tokens"],
                "temperature": config["temperature"]
            }
    }) for i, prompt in enumerate(reqs)]

    batch_file = BytesIO('\n'.join(batch_reqs).encode())
    batch_input_file = openai.files.create(file=batch_file, purpose="batch")

    batch_metadata = openai.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=config
    )

    return batch_metadata
