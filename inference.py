
# Commented out IPython magic to ensure Python compatibility.
# pip install llama-index
# pip install transformers accelerate bitsandbytes
# pip install llama-index-readers-web
# pip install llama-index-llms-huggingface
# pip install llama-index-embeddings-huggingface
# pip install llama-index-program-openai
# pip install llama-index-agent-openai
# pip install pdfminer.six

from pdfminer.high_level import extract_text
from llama_index.core import Document
import torch
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

MAX_LENGTH = 2048

MODEL_DIRECTORY_MAP = {
    "gemma-2" : "google/gemma-1.1-7b-it",
    "llama2-13b": "/raid/ganesh/nagakalyani/Downloads/Llama-2-13b-chat-hf",
}

DEFAULT_SYSTEM_PROMPT = "You are an experienced doctor and give honest assistant to user when a query is asked. Answer the question based on the context below. Keep the answer short. Respond 'Unsure about answer' if not sure about the answer."


def extract_pdf_text(pdf_path):
    with open(pdf_path, 'rb') as f:
        raw_text = extract_text(f)
    return raw_text.strip()


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def initialize_model_and_tokenizer(model_type="gemma-2", device_map=None):
    start_time = time.time()

    model_directory_path = MODEL_DIRECTORY_MAP[model_type]
    tokenizer = AutoTokenizer.from_pretrained(model_directory_path)
    tokenizer.pad_token = tokenizer.eos_token

    if device_map is None:
        device_map = {"": "cuda:1"}

    model = AutoModelForCausalLM.from_pretrained(
        model_directory_path, device_map=device_map
    ).eval()
    tokenizer.padding_side = "right"

    end_time = time.time()
    print(f"Loaded model and tokenizer in {end_time - start_time} seconds")

    return tokenizer, model


if __name__ == "__main__":

    tokenizer, model = initialize_model_and_tokenizer()
    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        query_wrapper_prompt=PromptTemplate(
            "<s> [INST] You are an experienced doctor and give honest assistant to user when a query is asked. Answer the question based on the context below. Keep the answer short. Respond 'Unsure about answer' if not sure about the answer. \t\t\t{query_str} [/INST] "),
        context_window=3900,
        model_kwargs={"quantization_config": quantization_config}
    )
    Settings.llm = llm
    Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

    print("Model and tokenizer initialized. Waiting for prompts...")

    # doc_path = input(
    #     "Enter the path of the text file path or type '0' if there is no path: ")
    doc_path = "/raid/ganesh/vishak/ashutosh/COE/data/Patient_1_Discharge summary_Final.pdf"
    text = extract_pdf_text(doc_path)
    documents = [Document(text=text)]
    vector_index = VectorStoreIndex.from_documents(documents)
    query_engine = vector_index.as_query_engine(response_mode="compact")
    
    print(documents)
    exit()

    history = []
    add_system_prompt = True

    vector_index_results = open('vector_index_results', 'a')

    while True:
        prompt = input(
            "Enter your prompt or type 'exit' to quit or 'restart' to restart the conversation: ")

        if prompt.lower() == 'exit':
            break
        elif prompt.lower() == 'restart':
            history.clear()
            add_system_prompt = True
            print("Conversation restarted. History cleared.")
            continue
        else:
            response = query_engine.query(prompt)
            print(response)
            vector_index_results.write(
                "\n ------------------------------------------------------------------------------------------------------ \n")

            vector_index_results.write(
                "Query : \n" + prompt + "\n\nResponse: \n" + str(response))
            vector_index_results.close()

    history.extend([
        {'role': 'assistant', 'content': response}
    ])
