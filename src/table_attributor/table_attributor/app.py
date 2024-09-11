import streamlit as st
import pandas as pd
from main import get_attributed_image

st.write("Get Attribution from LLM Answer")

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""


uploaded_file = st.file_uploader("Choose your .pdf file", type="pdf")
answer = st.text_input(
    label="Enter LLM answer here",
    value="",
)
if len(answer):
    if uploaded_file:
        if (
            uploaded_file.name != st.session_state.pdf_filename
            or answer != st.session_state.answer
        ):
            with st.spinner("Thinking..."):
                final_image = get_attributed_image(
                    pdf_path=uploaded_file.name, answer=answer
                )
                st.session_state.answer = answer
                st.session_state.pdf_filename = uploaded_file.name
                st.image(final_image)
                st.session_state.pdf_uploaded = True
    else:
        st.error("Please provide a PDF !")
