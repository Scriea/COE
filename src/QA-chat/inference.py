from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from pdfminer.high_level import extract_text    
from llama_index.core import Document

MAX_LENGTH = 2048

MODEL_DIRECTORY_MAP = {
    # "7b": "<give path to 7b model if you want to after cloning from github repo>",
    "llama2-13b": "/raid/ganesh/nagakalyani/Downloads/Llama-2-13b-chat-hf",
    "mixtral_8x7B": "/raid/ganesh/nagakalyani/Downloads/Mixtral-8x7B-Instruct-v0.1"
}
DEFAULT_SYSTEM_PROMPT = " You are an expert in assisting question-answering task. Answer the question based on the context below. Keep the answer short. Respond 'Unsure about answer' if not sure about the answer."

# check the GPUs which are free and give it to intialize_model_and_tokenizer and generate_response functions before running inference


def initialize_model_and_tokenizer(model_type="llama2-13b", device_map=None):
    start_time = time.time()

    model_directory_path = MODEL_DIRECTORY_MAP[model_type]
    tokenizer = AutoTokenizer.from_pretrained(model_directory_path)
    tokenizer.pad_token = tokenizer.eos_token

    if device_map is None:
        device_map = {"": "cuda:6"}  # Default to one GPU

    model = AutoModelForCausalLM.from_pretrained(
        model_directory_path, device_map=device_map
    ).eval()
    tokenizer.padding_side = "right"

    end_time = time.time()
    print(f"Loaded model and tokenizer in {end_time - start_time} seconds")

    return tokenizer, model

def extract_pdf_text(pdf_path):
    with open(pdf_path, 'rb') as f:
        raw_text = extract_text(f)
    raw_text.strip()
    raw_text.replace(r'\n', '\n')
    #documents = [Document(text=raw_text.strip())]
    return raw_text


def format_prompt(history):
    formatted_prompt = ""
    new_start = True
    for message in history:
        if new_start:
            formatted_prompt += "<s>[INST] "
            new_start = False
        if message['role'] == 'system':
            formatted_prompt += f"<<SYS>>\\n{message['content']}\\n<</SYS>>\\n\\n"
        elif message['role'] == 'user':
            formatted_prompt += f"{message['content']}"
            formatted_prompt += " [/INST]"
        elif message['role'] == 'assistant':
            formatted_prompt += f" {message['content']}"
            formatted_prompt += " </s>"
            new_start = True
    return formatted_prompt


def generate_response(prompt, history, tokenizer, model, max_length=MAX_LENGTH, add_system_prompt=False):
    if add_system_prompt:
        history.extend([{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT}])
    history.extend([{'role': 'user', 'content': prompt}])
    formatted_prompt = format_prompt(history)


    inputs = tokenizer(formatted_prompt, return_tensors="pt",
                       truncation=True, max_length=max_length).to("cuda:6")
    output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.1,
        temperature=0.2,
        max_new_tokens=max_length
    )
    # Extract the new tokens (response) from the generated tokens.
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response


def read_file(file_path):
    with open(file_path, 'r') as f:
        data = f.read()

    return data


def prompt_with_data_and_query(file_path, query):
    data = extract_pdf_text(file_path)

#     prompt = f'''
# Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
# If you don't know the answer, just say that you don't know. Don't try to make up an answer.
# ALWAYS return a "SOURCES" part in your answer.

# ### INPUT DOCUMENT:
# {data}

# ### QUESTION:
# {query}

# '''
# Content: ...
# Source: ...
# ...
# =========
# FINAL ANSWER:
# SOURCES: 

    prompt = f'''
\n<</SYS>>\n\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\nGenerate the next agent response by answering the question. Answer it as succinctly as possible. If the answer comes from different documents please mention all possibilities in your answer and use the titles to separate between topics or domains. If you cannot answer the question from the given documents, please state that you do not have an answer.\n"""


Using this information : {data} answer the Question : {query}

'''

if __name__ == "__main__":

    # # use the device_map when one GPU is not sufficient
    # device_map = {
    #     "": "cuda:6",  # Use the first GPU
    #     "": "cuda:1"   # Use the second GPU
    # }

    tokenizer, model = initialize_model_and_tokenizer()

    print("Model and tokenizer initialized. Waiting for prompts...")

    doc_path = input(
        "Enter the path of the text file path or type '0' if there is no path: ")

    history = []
    add_system_prompt = True

    while True:
        query = input(
            "Enter your query or type 'exit' to quit or 'restart' to restart the conversation: ")

        if (doc_path != '0' and query.lower() != 'exit' and query.lower != 'restart'):
            prompt = prompt_with_data_and_query(doc_path, query)
        else:
            prompt = query

        if prompt.lower() == 'exit':
            break
        elif prompt.lower() == 'restart':
            history.clear()
            add_system_prompt = True
            print("Conversation restarted. History cleared.")
            continue
        time1 = time.time()
        response = generate_response(
            prompt, history, tokenizer, model, max_length=MAX_LENGTH, add_system_prompt=add_system_prompt)
        add_system_prompt = False
        save_response = open('saved_prompt','a')

        save_response.write("\n ------------------------------------------------------------------------------------------------------ \n")
        save_response.write("Prompt : \n" + prompt)
        save_response.write("Query : \n" + query + "\n\nResponse: \n" + response)
        save_response.close()
        print("Query : \n", query)
        print("\n Response: \n",response)
        time2 = time.time()
        print(" \n The time taken to generate response is ", time2-time1)

        history.extend([
            {'role': 'assistant', 'content': response}
        ])
