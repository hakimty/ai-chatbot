import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

st.set_page_config(page_title="AI Conversational Assistant", page_icon="🤖")
st.title("🤖 AI Conversational Assistant")

# Safely fetch API key from Streamlit Secrets or local .env
api_key = None

try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

# Sidebar Diagnostics & Controls
with st.sidebar:
    st.header("⚙️ App Status")
    
    if api_key:
        # Hide key characters for security
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "Found"
        st.success(f"Key Found: `{masked_key}`")
    else:
        st.error("❌ API Key Missing")

    st.markdown("---")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Stop execution if no API key is present
if not api_key:
    st.warning("⚠️ **API Key Not Found!**")
    st.info("Please go to **Streamlit Cloud Dashboard → App Settings → Secrets** and add:\n\n`GEMINI_API_KEY = \"your_api_key_here\"`")
    st.stop()

# Initialize Gemini Client
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Gemini Client: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
if prompt := st.chat_input("Ask Nexus anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert chat history to SDK format
    contents = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # Stream AI Response
    with st.chat_message("assistant"):
        def generate_response_stream():
            try:
                response = client.models.generate_content_stream(
                    model="gemini-2.0-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction="You are Nexus, a helpful AI assistant.",
                        temperature=temperature,
                    ),
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            except Exception as err:
                yield f"⚠️ **API Error:** {err}"

        full_response = st.write_stream(generate_response_stream())

    st.session_state.messages.append({"role": "assistant", "content": full_response})