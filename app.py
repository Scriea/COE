import os
import subprocess
import json
import torch
import streamlit as st
import numpy as np
from src.attribution.attrb import AttributionModule
from src.generator.generator import Generator
from src.generator import prompts

ROOT_DIR = subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
chat_model = "meta-llama/Llama-2-7b-chat-hf" 
attribution_model = "meta-llama/Llama-2-7b-chat-hf" 

@st.cache_resource
def load_attribution_module():
    return AttributionModule(device="cuda:7")

@st.cache_resource
def load_generator():
    return Generator(model_name=chat_model, device="cuda:6")

# Initialize modules
attribution_module = load_attribution_module()
generator = load_generator()

# Streamlit app layout
st.title("Medical Agent")

# Input for user query
user_query = st.text_input("Enter your medical query:")

passages_file = os.path.join(ROOT_DIR, "data", "passages.json")
passages = attribution_module.load_paragraphs(passages_file=passages_file)
passages = "\n".join(passages)

embedding_file_path = os.path.join(attribution_module.output_dir, "paragraph_embeddings.npz")
search_index, paragraphs = attribution_module.create_faiss_index(embedding_file_path=embedding_file_path, ngpu=1)

if st.button("Get Response"):
    if user_query:
        # Generate response
        prompt = prompts.medical_prompt.format(passages, user_query)
        print(prompt)
        response = generator.generate_response(prompt)
        
        st.subheader("Generated Response")
        st.write(response)
    
        # Retrieve relevant paragraphs - Attribution Module
        retrieval_results = attribution_module.retrieve_paragraphs([user_query], search_index, paragraphs, k=1)
        retrieved_passages = retrieval_results[0]['retrieved_paragraphs'][0]
        st.subheader("Attribution")
        st.write(retrieved_passages)
    else:
        st.error("Please enter a query.")
