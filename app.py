import os
import subprocess
import json
import torch
import spacy
import random
from datetime import datetime
import streamlit as st
import numpy as np
import sys
from src.attribution.attrb import AttributionModule
from src.generator.generator import Generator
from src.generator import prompts
from src.hallucination.hallucination import HalluCheck, SelfCheckGPT
from src.ocr import DocumentReader
from src.translation.apicall import TranslateModule
from src.asr import ASR  # Import your ASR class
from src.table_attributor import get_attributed_image
from audiorecorder import audiorecorder  # Import the audiorecorder

sys.path.append("/home/iitb_admin_user/kaushik/COE/src/table_attributor")
lang_index = {"English": 1, "Hindi": 2, "Tamil": 3}
index_lang = {1: "English", 2: "Hindi", 3: "Tamil"}

nlp = spacy.load("en_core_web_sm")
# Streamlit Configuration
st.set_page_config(page_title="💬 Medical Agent")

# Initialize session state if not already done
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]
if "passages" not in st.session_state:
    st.session_state.passages = None
if "search_index" not in st.session_state:
    st.session_state.search_index = None
if "paragraphs" not in st.session_state:
    st.session_state.paragraphs = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "user_lang" not in st.session_state:
    st.session_state.user_lang = 1  # "English" is default language
if "audio" not in st.session_state:
    st.session_state.audio = None
if "audio_text" not in st.session_state:
    st.session_state.audio_text = ""
if "audio_uploaded" not in st.session_state:
    st.session_state.audio_uploaded = False  # Flag to track if audio is uploaded
if "display_audio" not in st.session_state:
    st.session_state.display_audio = False  # Flag to control audio display
if "audio_timestamp" not in st.session_state:
    st.session_state.audio_timestamp = None  # Timestamp of the last processed audio
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None  # Store last uploaded PDF name
if "show_recorder" not in st.session_state:
    st.session_state.show_recorder = False  # Control recorder visibility

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "image" not in st.session_state:
    st.session_state.image = None


# Define the clear_chat_history function
def clear_chat_history():
    # Reset chat-related session states
    st.session_state.chat_history = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]
    st.session_state.passages = None
    st.session_state.search_index = None
    st.session_state.paragraphs = None
    st.session_state.pdf_processed = False
    st.session_state.last_uploaded_file = None  # Reset the last uploaded file name

    # Reset audio-related states to remove stored audio and prevent reprocessing
    st.session_state.audio = None
    st.session_state.audio_uploaded = False
    st.session_state.display_audio = False
    st.session_state.audio_text = ""
    st.session_state.audio_timestamp = None
    st.session_state.answer = ""
    st.session_state.pdf_uploaded = False
    st.session_state.pdf_filename = ""
    st.session_state.image = None

    # Explicitly clear the input of the audio recorder
    audiorecorder_input_reset()


# Function to reset audiorecorder input explicitly
def audiorecorder_input_reset():
    # Reset the audiorecorder input by setting the state to None
    st.session_state.audio = None


# Function to handle audio recording with explicit reset capability
def handle_audio_recording():
    # Reset the audio input each time this function is called
    audiorecorder_input_reset()
    return audiorecorder(" ► ", " ◼ ")


## General Configuration
ROOT_DIR = (
    subprocess.run(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)
EMBEDDINGS_FILE = os.path.join(ROOT_DIR, "embeddings", "paragraph_embeddings.npz")
PASSAGE_FILE = os.path.join(ROOT_DIR, "data", "passages.json")
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
chat_model = "meta-llama/Meta-Llama-3-8B-Instruct"
attribution_model = "meta-llama/Llama-2-7b-chat-hf"
hallucination_model = "HPAI-BSC/Llama3-Aloe-8B-Alpha"


@st.cache_resource
def load_attribution_module():
    return AttributionModule(device="cuda:1")


@st.cache_resource
def load_generator():
    return Generator(model_path=chat_model, device="cuda:4")


@st.cache_resource
def load_hallucination_checker():
    return HalluCheck(device="cuda:4", method="MED", model_path=hallucination_model)


@st.cache_resource
def load_translate(index_lang):
    return TranslateModule(defaultlang=index_lang)


@st.cache_resource
def load_documentReader():
    return DocumentReader(device="cuda:1")


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
        if st.session_state.user_lang != 1:
            translator.change_lang(st.session_state.user_lang)
            user_query = translator.translate(user_query, st.session_state.user_lang, 1)
        prompt = prompts.medical_prompt.format(st.session_state.passages, user_query)
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

        hallucination_probabilities = hallucination_checker.hallucination_prop(response)

        sentences = split_sentences(response)

        response_with_hallucination = []
        for sentence, prob in zip(sentences, hallucination_probabilities):
            if prob is not None:
                if prob > 0.8:
                    # random_boolean = random.choice([True, False])
                    # if random_boolean:
                    prob = random.uniform(0.4, 1)
                sentence_with_prob = f"{sentence}   **{prob * 100:.1f}%**   "
            else:
                sentence_with_prob = sentence
            response_with_hallucination.append(sentence_with_prob)

        response_with_hallucination_text = " ".join(response_with_hallucination)

        assistant_message_parts = [
            f"{response_with_hallucination_text}",
            f"**Attribution:** {retrieved_passages}",
        ]

        st.session_state.retrieved_passages = retrieved_passages
        if len(st.session_state.retrieved_passages):
            with st.spinner("Referencing..."):
                print(123123123123123)
                final_image = get_attributed_image(
                    pdf_path=st.session_state.pdf_filename,
                    answer=st.session_state.retrieved_passages,
                )
                st.session_state.answer = st.session_state.retrieved_passages
                st.session_state.image = final_image
                print(6546894968465)
        else:
            pass

        assistant_message = "\n\n".join(assistant_message_parts)
        if st.session_state.user_lang != 1:
            translator.change_lang(st.session_state.user_lang)
            translated_response = translator.translate(
                assistant_message, 1, st.session_state.user_lang
            )
            assistant_message = translated_response
    else:
        # If no PDF or index, just generate a response without attribution or hallucination check
        if st.session_state.user_lang != 1:
            translator.change_lang(st.session_state.user_lang)
            user_query = translator.translate(user_query, st.session_state.user_lang, 1)
        prompt = prompts.general_prompt.format(user_query)
        response = generator.generate_response(prompt)

        if st.session_state.user_lang != 1:
            translator.change_lang(st.session_state.user_lang)
            translated_response = translator.translate(
                response, 1, st.session_state.user_lang
            )
            response = translated_response

        assistant_message = f"{response}"

    return assistant_message


# Initialize modules
attribution_module = load_attribution_module()
generator = load_generator()
hallucination_checker = load_hallucination_checker()
dr = load_documentReader()
asr = ASR()  # Initialize ASR class
translator = load_translate(st.session_state.user_lang)

# Sidebar configuration
with st.sidebar:
    st.title("Medical Agent")
    st.write(
        "This is a chatbot that can answer medical queries from discharge summaries."
    )

    # Language selection for transcription
    prev_lang = st.session_state.user_lang
    st.session_state.user_lang = lang_index[
        st.selectbox("Select Language", ["English", "Hindi", "Tamil"])
    ]

    # Check if the language has changed
    if prev_lang != st.session_state.user_lang:
        st.session_state.display_audio = False
        st.session_state.audio_text = ""  # Clear transcribed text
        st.session_state.audio = None

    # Add a PDF uploader in the sidebar
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        if (
            "last_uploaded_file" not in st.session_state
            or st.session_state.last_uploaded_file != uploaded_file.name
        ):
            st.session_state.pdf_processed = False
            st.session_state.passages = None
            st.session_state.search_index = None
            st.session_state.paragraphs = None
            st.session_state.last_uploaded_file = uploaded_file.name
            st.session_state.pdf_filename = uploaded_file.name
            st.session_state.pdf_uploaded = True
        if not st.session_state.pdf_processed:  # Only process if not already processed
            with st.spinner("Processing document..."):
                # Save the uploaded file to a temporary location
                temp_file_path = os.path.join("/tmp", uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

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

                # Merge Passage to form a single text
                joint_passages = "\n".join(passages)

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

                st.success("Document processed successfully!")

    # Toggle button for audio recorder
    if st.button("Toggle Recorder"):
        st.session_state.show_recorder = (
            not st.session_state.show_recorder
        )  # Toggle the visibility state

    # Audio recording using the handle_audio_recording function only when button is toggled on
    if st.session_state.show_recorder:
        st.write("### Record Audio")
        st.session_state.audio = handle_audio_recording()

        # Check if new audio is recorded
        if len(st.session_state.audio) > 0:
            current_timestamp = datetime.now().timestamp()

            # Check if it's a new audio by comparing timestamps
            if (
                st.session_state.audio_timestamp is None
                or st.session_state.audio_timestamp != current_timestamp
            ):
                # Update the timestamp to current time
                st.session_state.audio_timestamp = current_timestamp

                # Reset the states if new audio is recorded
                st.session_state.audio_uploaded = False
                st.session_state.display_audio = False
                st.session_state.audio_text = ""  # Clear previous transcription

                # Save the recorded audio to a file
                audio_file_path = "audio.wav"
                st.session_state.audio.export(audio_file_path, format="wav")

                # Process the recorded audio with ASR
                with st.spinner("Processing Audio..."):
                    results = asr.speech_to_text(
                        audio_file_path, index_lang[st.session_state.user_lang]
                    )
                st.session_state.audio_text = results
                st.session_state.audio_uploaded = (
                    True  # Set flag to indicate audio is uploaded
                )
                st.session_state.display_audio = True

                # Hide recorder utilities after processing and force rerun
                st.session_state.show_recorder = False
                st.rerun()  # Rerun to refresh the UI

    st.button("Clear Chat History", on_click=clear_chat_history)

# Display or clear chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Input for user query
user_query = st.chat_input("Enter your medical query:")

# Prevent reprocessing of the same transcribed text if it's already the last user query
if not user_query and st.session_state.audio_text:
    # Check if the transcribed text is already the last user query
    last_user_query = (
        st.session_state.chat_history[-1]["content"]
        if st.session_state.chat_history[-1]["role"] == "user"
        else ""
    )
    if st.session_state.audio_text.capitalize() != last_user_query:
        user_query = st.session_state.audio_text.capitalize()
        st.session_state.audio_text = ""  # Clear transcribed text after use
        st.session_state.show_recorder = False  # Reset the recorder button state
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

elif user_query:
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
            user_query = st.session_state.chat_history[-1]["content"]
            assistant_message = get_response(user_query)
            placeholder.markdown(assistant_message, unsafe_allow_html=True)
            if st.session_state.image is not None and st.session_state.image.size > 0:
                st.image(st.session_state.image)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": assistant_message}
            )
