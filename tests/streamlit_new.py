import os
import subprocess
import streamlit as st
import re
import sys

ROOT_DIR = subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
sys.path.append(ROOT_DIR)

from src.generator import prompts

# Mock functions for testing without GPU
class AttributionModule:
    def retrieve_paragraphs(self, queries, search_index, paragraphs, k=1):
        return [{'retrieved_paragraphs': ['Mocked relevant passage for testing.']}]

class Generator:
    def generate_response(self, input_text):
        return "This is a test response. It has multiple sentences. Some of these might be hallucinations."

class HalluCheck:
    def hallucination_prop(self, response, context=None):
        return [0.1, None, 0.5]  # Mocked probabilities for testing

# Streamlit Configuration
st.set_page_config(page_title="💬 Medical Agent")

# Input for user query
with st.sidebar:
    st.title("Medical Agent")
    st.write("This is a chatbot that can answer medical queries from discharge summaries.")

## General Configuration
ROOT_DIR = subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")

# Load the mocked modules
@st.cache_resource
def load_attribution_module():
    return AttributionModule()

@st.cache_resource
def load_generator():
    return Generator()

@st.cache_resource
def load_hallucination_checker():
    return HalluCheck()

def split_sentences(text):
    # This function splits the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def get_response(user_query):
    # Generate response
    prompt = prompts.medical_prompt.format(joint_passages, user_query)
    response = generator.generate_response(prompt)
    
    # Attribution
    attribution_query = f"Question: {user_query}\nAnswer: {response}"
    # Retrieve relevant paragraphs - Attribution Module
    retrieval_results = attribution_module.retrieve_paragraphs([attribution_query], search_index, paragraphs, k=1)
    retrieved_passages = retrieval_results[0]['retrieved_paragraphs'][0]

    # Check for hallucination
    hallucination_probabilities = hallucination_checker.hallucination_prop(response, context="joint_passages")

    # Split response into sentences
    sentences = split_sentences(response)
    
    # Compile the response with hallucination probabilities
    response_with_hallucination = []
    for sentence, prob in zip(sentences, hallucination_probabilities):
        if prob is not None:
            sentence_with_prob = f"{sentence} <span style='background-color: yellow;'>{prob * 100:.2f}%</span>"
        else:
            sentence_with_prob = sentence
        response_with_hallucination.append(sentence_with_prob)
    
    response_with_hallucination_text = " ".join(response_with_hallucination)

    # Compile the response message
    assistant_message_parts = [
        f"**Generated Response:** {response_with_hallucination_text}",
        f"**Attribution:** {retrieved_passages}"
    ]
    assistant_message = "\n\n".join(assistant_message_parts)
    return assistant_message

# Initialize modules
attribution_module = load_attribution_module()
generator = load_generator()
hallucination_checker = load_hallucination_checker()

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

## Processing Document
joint_passages = """
    ADVICE @ DISCHARGE :
    1. Regular medications & STOP SMOKING.
    2. Avoid Alcohol, Heavy exertion and lifting weights.
    3. Diet - High fiber, low cholesterol, low sugar (no sugar if diabetic), fruits, vegetables (5 servings per day).
    4. Exercise - Walk at least for 30 minutes daily. Avoid if Chest pain.
    5. TARGETS * LDL<70mg/dl *BP - 120/80mmHg * Sugar Fasting - 100mg/dl Post Breakfast – 150mg/dl * BMI<25kg/m2.
    6. IF CHEST PAIN – T.ANGISED 0.6 mg or T.SORBITRATE 5 mg keep under tongue. Repeat if no relief @ 5 minutes and report to nearest doctor for urgent ECG.
"""

# Mocking embeddings and search index for testing
search_index = None
paragraphs = None

# Add a clear history button
def clear_chat_history():
    st.session_state.chat_history = [{"role": "assistant", "content": "How may I assist you today?"}]

st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

# Input for user query
user_query = st.chat_input("Enter your medical query:")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

# Generate a new response if the last message is not from the assistant
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            placeholder = st.empty()
            
            assistant_message = get_response(st.session_state.chat_history[-1]["content"])
            placeholder.markdown(assistant_message, unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
