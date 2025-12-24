import os
from pathlib import Path

import dotenv
from openai import OpenAI

dotenv.load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

clarifai_pat = (
    os.getenv("CLARIFAI_PAT")
    or os.getenv("CLARIFAI_API_KEY")
    or os.getenv("CLARIFAI_TOKEN")
)
if not clarifai_pat:
    raise RuntimeError(
        "Missing Clarifai credentials. Set `CLARIFAI_PAT` (preferred) or "
        "`CLARIFAI_API_KEY`/`CLARIFAI_TOKEN` in your environment or in `.env`."
    )

client = OpenAI(
    base_url="https://api.clarifai.com/v2/ext/openai/v1",
    api_key=clarifai_pat,
)
response = client.chat.completions.create(
    model="https://clarifai.com/deepseek-ai/deepseek-ocr/models/DeepSeek-OCR/versions/86b122666c2548f88d04dd998ccfbd70",
    messages=[
        {"role": "system", "content": "Talk like a pirate."},
        {
            "role": "user",
            "content": "How do I check if a Python object is an instance of a class?",
        },
    ],
    temperature=0.7,
    stream=False, # stream=True also works, just iterator over the response
)
print(response)
