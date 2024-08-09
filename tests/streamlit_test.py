import streamlit as st

# Streamlit Configuration
st.set_page_config(page_title="💬 Medical Agent")

# Input for user query
with st.sidebar:
    st.title("Medical Agent")
    st.write("This is a chatbot that can answer medical queries from discharge summaries.")

# Temporary test data
def get_response(user_query):
    # Temporary response and test data
    response = "This is a generated response. It is intended to simulate the output of the real system. The server is currently busy, the real system has been shut down!!"
    retrieved_passages = "This is a retrieved passage. It simulates the attribution module output."
    hallucination_probabilities = [0.1, 0.2, 0.3]  # Example probabilities for testing

    # Compile the response message
    hallucination_probabilities_str = ', '.join([f"{prob * 100:.2f}%" if prob is not None else "N/A" for prob in hallucination_probabilities])
    assistant_message_parts = [
        f"**Generated Response:** {response}",
        f"**Attribution:** {retrieved_passages}",
        f"**Hallucination Probabilities:** {hallucination_probabilities_str}"
    ]
    assistant_message = "\n\n".join(assistant_message_parts)
    return assistant_message

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

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
