from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from collections import defaultdict
import os

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")

client = InferenceClient(
    provider="novita",
    api_key=HF_TOKEN,
)

# FastAPI app
app = FastAPI()

# Define the default system prompt
DEFAULT_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are PoultryLlama, a friendly, knowledgeable assistant for poultry farmers. "
        "Your job is to provide clear, practical, and trustworthy advice on raising healthy poultry. "
        "Always explain concepts in simple terms, avoid jargon, and focus on what a farmer can do realistically. "
        "You can answer questions about chicken health, feeding, diseases, housing, egg production, and more. "
        "Be empathetic, supportive, and non-judgmental — many users may be new or facing challenges. "
        "Avoid speculation, and if something is uncertain or needs a vet's attention, say so politely."
    )
}

# Store conversation history per user with system prompt
conversation_history = defaultdict(lambda: {"history": [DEFAULT_SYSTEM_PROMPT.copy()]})

class ChatInput(BaseModel):
    user_id: str
    message: str

@app.post("/chat/")
async def chat(input_data: ChatInput):
    user_id = input_data.user_id
    user_msg = input_data.message

    history = conversation_history[user_id]["history"]
    history.append({"role": "user", "content": user_msg})

    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=history,
        )
        assistant_msg = response.choices[0].message
        history.append({"role": "assistant", "content": assistant_msg["content"]})
        return {"message": assistant_msg["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Welcome to PoultryLlama Chat API.",
        "endpoints": {
            "/chat/": "Communicate with the LLM to understand more about poultry farming.",
        },
    }
