from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import subprocess
import torch
import spacy
from src.attribution.attrb import AttributionModule
from src.generator.generator import Generator
from src.generator import prompts
from src.hallucination.hallucination import HalluCheck

# Initialize FastAPI
app = FastAPI(
    title="Medical Agent API",
    description="APIs for Medical Chatbot with Hallucination and Attribution Modules",
    version="1.0.0"
)

# Load NLP model for sentence splitting
nlp = spacy.load("en_core_web_sm")

# General Configuration
ROOT_DIR = subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
chat_model = "meta-llama/Llama-2-7b-chat-hf"
attribution_model = "meta-llama/Llama-2-7b-chat-hf"
hallucination_model = "HPAI-BSC/Llama3-Aloe-8B-Alpha"

# Load the required modules
attribution_module = AttributionModule(device="cuda:2")
generator = Generator(model_path=chat_model, device="cuda:2")
hallucination_checker = HalluCheck(device="cuda:5", method="MED", model_path=hallucination_model)

# Load passages
passages_file = os.path.join(ROOT_DIR, "data", "passages.json")
passages = attribution_module.load_paragraphs(passages_file=passages_file)
joint_passages = "\n".join(passages)

embedding_file_path = os.path.join(attribution_module.output_dir, "paragraph_embeddings.npz")
if not os.path.exists(embedding_file_path):
    attribution_module.vectorize_paragraphs(passages_file, embedding_file_path)

search_index, paragraphs = attribution_module.create_faiss_index(embedding_file_path=embedding_file_path, ngpu=1)

# Define the request and response models
class QueryRequest(BaseModel):
    user_query: str

    class Config:
        schema_extra = {
            "example": {
                "user_query": "What is hypertension?"
            }
        }

class ChatResponse(BaseModel):
    response: str
    answer: str
    attribution: str
    hallucinations: List[float]

    class Config:
        schema_extra = {
            "example": {
                "response": "Hypertension, also known as high blood pressure, is a medical condition where the blood pressure in the body is consistently too high. <span style='background-color: yellow;'>100.00%</span> This can lead to damage of blood vessels...",
                "answer": "Hypertension, also known as high blood pressure, is a medical condition where the blood pressure in the body is consistently too high. This can lead to damage of blood vessels...",
                "attribution": "RISK FACTORS: Hypertension, Diabetes",
                "hallucinations": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }
        }

class ClearHistoryResponse(BaseModel):
    message: str

    class Config:
        schema_extra = {
            "example": {
                "message": "Chat history cleared"
            }
        }

class ResponseItem(BaseModel):
    role: str
    content: str

class ChatHistory(BaseModel):
    chat_history: List[ResponseItem]

    class Config:
        schema_extra = {
            "example": {
                "chat_history": [
                    {"role": "assistant", "content": "How may I assist you today?"},
                    {"role": "user", "content": "What is hypertension?"},
                    {"role": "assistant", "content": "Hypertension, also known as high blood pressure, is a condition..."}
                ]
            }
        }

# Helper functions
def split_sentences(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def get_response(user_query):
    prompt = prompts.medical_prompt.format(joint_passages, user_query)
    response = generator.generate_response(prompt)

    attribution_query = f"Question: {user_query}\nAnswer: {response}"
    retrieval_results = attribution_module.retrieve_paragraphs([attribution_query], search_index, paragraphs, k=1)
    retrieved_passages = retrieval_results[0]['retrieved_paragraphs'][0]

    hallucination_probabilities = hallucination_checker.hallucination_prop(response, context="joint_passages")

    sentences = split_sentences(response)

    response_with_hallucination = []
    for sentence, prob in zip(sentences, hallucination_probabilities):
        if prob is not None:
            sentence_with_prob = f"{sentence} <span style='background-color: yellow;'>{prob * 100:.2f}%</span>"
        else:
            sentence_with_prob = sentence
        response_with_hallucination.append(sentence_with_prob)

    response_with_hallucination_text = " ".join(response_with_hallucination)

    # Create the structured response
    structured_response = {
        "response": response_with_hallucination_text,
        "answer": response,
        "attribution": retrieved_passages,
        "hallucinations": hallucination_probabilities
    }

    return structured_response

# API Endpoints

# Chat endpoint
@app.post("/chat", summary="Get response from the chatbot", response_model=ChatResponse)
def chat(request: QueryRequest):
    try:
        user_query = request.user_query
        structured_response = get_response(user_query)
        return structured_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Clear chat history endpoint
@app.post("/clear", summary="Clear the chat history", response_model=ClearHistoryResponse)
def clear_chat_history():
    # Here, you would clear the chat history from wherever it's stored (e.g., a database)
    # For now, we'll return a mock response
    return {"message": "Chat history cleared"}

# Get chat history endpoint
@app.post("/history", summary="Get chat history", response_model=ChatHistory)
def get_chat_history():
    # Here, you would retrieve the chat history from storage (e.g., a database)
    # For now, we'll return a mock chat history
    chat_history = [
        {"role": "assistant", "content": "How may I assist you today?"},
        {"role": "user", "content": "What is hypertension?"},
        {"role": "assistant", "content": "Hypertension, also known as high blood pressure, is a condition..."}
    ]
    return {"chat_history": chat_history}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)