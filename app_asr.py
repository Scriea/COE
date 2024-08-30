import os
import subprocess
import json
import torch
import spacy
import streamlit as st
import numpy as np
from src.attribution import AttributionModule
from src.generator.generator import Generator
from src.generator import prompts
from src.hallucination.hallucination import HalluCheck, SelfCheckGPT
from src.ocr import DocumentReader
from src.asr import ASR  # Import your ASR class
from audiorecorder import audiorecorder  # Import the audiorecorder


lang_index = {
    "English": 1,
    "Hindi": 2,
    "Tamil": 3
}

index_lang = {
    1: "English",
    2: "Hindi",
    3: "Tamil"
}

nlp = spacy.load("en_core_web_sm")
# Streamlit Configuration
st.set_page_config(page_title="💬 Medical Agent")

# Initialize session state if not already done
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "How may I assist you today?"}]
if "passages" not in st.session_state:
    st.session_state.passages = None
if "search_index" not in st.session_state:
    st.session_state.search_index = None
if "paragraphs" not in st.session_state:
    st.session_state.paragraphs = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "audio_text" not in st.session_state:
    st.session_state.audio_text = ""
if "user_lang" not in st.session_state:
    st.session_state.user_lang = 1 # "English" is default language

# Define the clear_chat_history function
def clear_chat_history():
    st.session_state.chat_history = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.passages = None
    st.session_state.search_index = None
    st.session_state.paragraphs = None
    st.session_state.pdf_processed = False
    st.session_state.audio_text = ""  # Reset the audio text

## General Configuration
ROOT_DIR = subprocess.run(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE).stdout.decode("utf-8").strip()
EMBEDDINGS_FILE = os.path.join(ROOT_DIR, "embeddings", "paragraph_embeddings.npz")
PASSAGE_FILE = os.path.join(ROOT_DIR, "data", "passages.json")
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
chat_model = "meta-llama/Meta-Llama-3-8B-Instruct"
attribution_model = "meta-llama/Llama-2-7b-chat-hf"
hallucination_model = "HPAI-BSC/Llama3-Aloe-8B-Alpha"

@st.cache_resource
def load_attribution_module():
    return AttributionModule(device="cuda:2")

@st.cache_resource
def load_generator():
    return Generator(model_path=chat_model, device="cuda:2")

@st.cache_resource
def load_hallucination_checker():
    return HalluCheck(device="cuda:4", method="MED", model_path=hallucination_model)

def split_sentences(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def get_response(user_query):
    if (
        st.session_state.passages is not None
        and st.session_state.search_index is not None
        and st.session_state.paragraphs is not None
        and st.session_state.paragraphs.size > 0
    ):
        prompt = prompts.medical_prompt.format(st.session_state.passages, user_query)
        print(prompt)
        response = generator.generate_response(prompt)
        attribution_paragraphs = [""]
        attribution_query = f"Question: {user_query}\nAnswer: {response}"
        retrieval_results = attribution_module.retrieve_paragraphs(
            [attribution_query],
            st.session_state.search_index,
            st.session_state.paragraphs,
            k=1,
        )
        retrieved_passages = retrieval_results[0]["retrieved_paragraphs"][0]

        hallucination_probabilities = hallucination_checker.hallucination_prop(
            response, 
            # context=st.session_state.passages
        )

        sentences = split_sentences(response)

        response_with_hallucination = []
        for sentence, prob in zip(sentences, hallucination_probabilities):
            if prob is not None:
                sentence_with_prob = f"{sentence} <span style='background-color: yellow;'>{prob * 100:.2f}%</span>"
            else:
                sentence_with_prob = sentence
            response_with_hallucination.append(sentence_with_prob)

        response_with_hallucination_text = " ".join(response_with_hallucination)

        assistant_message_parts = [
            f"**Generated Response:** {response_with_hallucination_text}",
            f"**Attribution:** {retrieved_passages}",
        ]
        assistant_message = "\n\n".join(assistant_message_parts)
    else:
        # If no PDF or index, just generate a response without attribution or hallucination check
        prompt = prompts.general_prompt.format(user_query)
        response = generator.generate_response(prompt)
        assistant_message = f"**Generated Response:** {response}"

    return assistant_message

# Initialize modules
attribution_module = load_attribution_module()
generator = load_generator()
hallucination_checker = load_hallucination_checker()
asr = ASR()  # Initialize ASR class

# Sidebar configuration
with st.sidebar:
    st.title("Medical Agent")
    st.write("This is a chatbot that can answer medical queries from discharge summaries.")

    # Language selection for transcription
    st.session_state.user_lang = lang_index[st.selectbox("Select Language for Transcription", ["English", "Hindi", "Tamil", "Telugu", "Malayalam"])]

    # Add a PDF uploader in the sidebar
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        if not st.session_state.pdf_processed:  # Only process if not already processed
            with st.spinner("Processing document..."):
                # Save the uploaded file to a temporary location
                temp_file_path = os.path.join("/tmp", uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Initialize the DocumentReader
                dr = DocumentReader(device="cuda:4")

                # Extract text from the uploaded PDF
                extracted_text = dr.extract_text(temp_file_path)

                # Extract specific passages from the uploaded PDF
                phrases = [
                    "Discharge Summary",
                    "HISTORY",
                    "RISK FACTORS",
                    "CLINICAL FINDINGS",
                    "ADMISSION DIAGNOSIS",
                    "PREV. INTERVENTION",
                    "PROCEDURES DETAILS",
                    "RESULT :",
                    "IN HOSPITAL COURSE :",
                    "FINAL DIAGNOSIS",
                    "CONDITION AT DISCHARGE",
                    "ADVICE @ DISCHARGE",
                    "DIET ADVICE",
                    "DISCHARGE MEDICATIONS",
                ]
                passages = dr.extract_passages(
                    temp_file_path, output_path=PASSAGE_FILE, phrases=phrases
                )
                st.session_state.passages = passages  # Store in session state

                ## Merge Passage to form a single text
                joint_passages = "\n".join(passages)
                st.success("Document processed successfully!")
                # st.write("**Extracted Text:**")
                # st.text(joint_passages)
                # st.write("**Extracted Passages:**")
                # st.json(passages)

                # Vectorize the passages extracted from OCR
                embedding_file_path = os.path.join(
                    attribution_module.output_dir, "paragraph_embeddings.npz"
                )
                attribution_module.vectorize_paragraphs(
                    passages_file=PASSAGE_FILE,
                    output_file=embedding_file_path,
                    force=True,
                )

                # Proceed with creating the FAISS index
                search_index, paragraphs = attribution_module.create_faiss_index(
                    embedding_file_path=embedding_file_path, ngpu=1
                )
                st.session_state.search_index = search_index  # Store in session state
                st.session_state.paragraphs = paragraphs  # Store in session state
                st.session_state.pdf_processed = (
                    True  # Set the flag to indicate processing is done
                )

    # Add a clear history button
    st.button("Clear Chat History", on_click=clear_chat_history)

# Audio recording
st.write("### Record Audio")
audio = audiorecorder(" ► ", " ◼ ", " || ")

if len(audio) > 0:
    st.audio(audio.export().read())  # Play the recorded audio

    # Save the recorded audio to a file
    audio_file_path = "audio.wav"
    audio.export(audio_file_path, format="wav")

    # Process the recorded audio with ASR
    results = asr.speech_to_text(audio_file_path, index_lang[st.session_state.user_lang])
    st.session_state.audio_text = results
    st.write(f"Transcribed Text: {results}")

# Display or clear chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input for user query
user_query = st.chat_input("Enter your medical query:")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

# Generate a new response if the last message is not from the assistant
if (
    st.session_state.chat_history is not None
    and st.session_state.chat_history[-1]["role"] != "assistant"
):
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            placeholder = st.empty()

            assistant_message = get_response(
                st.session_state.chat_history[-1]["content"]
            )
            placeholder.markdown(assistant_message, unsafe_allow_html=True)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": assistant_message}
            )
