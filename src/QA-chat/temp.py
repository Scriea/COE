from pdfminer.high_level import extract_text
from llama_index.core import Document
import json
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
    "llama2-13b": "/raid/ganesh/nagakalyani/Downloads/Llama-2-13b-chat-hf",
    "mixtral_8x7B": "/raid/ganesh/nagakalyani/Downloads/gemma-2-9b-it"
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


def initialize_model_and_tokenizer(model_type="mixtral_8x7B", device_map=None):
    start_time = time.time()

    model_directory_path = MODEL_DIRECTORY_MAP[model_type]
    tokenizer = AutoTokenizer.from_pretrained(model_directory_path)
    tokenizer.pad_token = tokenizer.eos_token

    if device_map is None:
        device_map = {"": "cuda:2"}

    model = AutoModelForCausalLM.from_pretrained(
        model_directory_path, device_map=device_map
    ).eval()
    tokenizer.padding_side = "right"

    end_time = time.time()
    print(f"Loaded model and tokenizer in {end_time - start_time} seconds")

    return tokenizer, model


if _name_ == "_main_":

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

    doc_paths = ["/raid/ganesh/nagakalyani/Downloads/input_pdf/Patient_1_Discharge summary_Final.pdf",
                 "/raid/ganesh/nagakalyani/Downloads/input_pdf/Patient_2_Discharge summary_Final.pdf", "/raid/ganesh/nagakalyani/Downloads/input_pdf/Patient_4_Discharge summary_Final.pdf"]
    # input(        "Enter the path of the text file path or type '0' if there is no path: ")
    results = {}
    for doc_path in doc_paths:
        # input(        "Enter the path of the text file path or type '0' if there is no path: ")

        text = extract_pdf_text(doc_path)
        documents = [Document(text=text)]
        # print(documents)
        vector_index = VectorStoreIndex.from_documents(documents)
        print(vector_index)
        query_engine = vector_index.as_query_engine(response_mode="compact")
        print(query_engine)

        history = []
        add_system_prompt = True
        final = {}

        vector_index_results = open('outputs/vsi_withoutcontextdict.txt', 'a')
        questions = [
            "What was the outcome of my angiography procedure?",
            "What medications should I take and how often?",
            "What diet should I follow to manage my condition?",
            "What are the targets I should aim for in terms of blood pressure, sugar levels, and BMI?",
            "What should I do if I experience chest pain?",
            "When is my follow-up appointment, and where should I go?",
            "What are the risks associated with my condition, and how can I prevent complications?",
            "What is the significance of my echocardiography report, and what does it mean for my condition?",
            "How long should I avoid heavy exertion and lifting weights?",
            "What are the next steps in my treatment plan, and what can I expect in the coming weeks/months?",
            "What medications do I need to continue taking, and what are their dosages?",
            "What lifestyle changes should I make to improve my heart health?"
        ]
        que = 0
        while que < 10:
            prompt = questions[que]
            # input(            "Enter your prompt or type 'exit' to quit or 'restart' to restart the conversation: ")

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
                final[que] = str(response)
                vector_index_results.write(
                    "\n ------------------------------------------------------------------------------------------------------ \n")

                vector_index_results.write(
                    "Query : \n" + prompt + "\n\nResponse: \n" + str(response))
                # vector_index_results.close()
            que += 1
        strfinal = str(final)
        results[f'gt{doc_path[-5]}'] = final
        strfinal = str(final)
        vector_index_results.write(strfinal)
        history.extend([
            {'role': 'assistant', 'content': response}
        ])
    with open('outputvsi.json', 'w') as f:
        json.dump(results, f, indent=4)