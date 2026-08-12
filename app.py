import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

load_dotenv()

st.set_page_config(page_title="AI Conversational Assistant", page_icon="🤖")
st.title("🤖 AI Conversational Assistant")

# Check for API Key
# Replace your current api_key line with this:
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Missing GEMINI_API_KEY in .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = "You are Nexus, a helpful, precise AI assistant."

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask Nexus anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # FIX: Keep only the last 10 messages to avoid token limit exhaustion
    MAX_HISTORY = 10
    recent_messages = st.session_state.messages[-MAX_HISTORY:]

    # Format history into SDK structure
    contents = []
    for msg in recent_messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # Stream assistant response safely
    with st.chat_message("assistant"):
        def generate_response_stream():
            try:
                response = client.models.generate_content_stream(

                    model="gemini-3-flash-preview", 
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=temperature,
                    ),
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            except APIError as e:
                # Catch 429 Quota Exceeded or other API errors
                if e.code == 429:
                    yield "⚠️ **Rate limit exceeded.** You have hit the API free-tier quota limit. Please wait about a minute and try again."
                else:
                    yield f"⚠️ **API Error:** {e.message}"

        full_response = st.write_stream(generate_response_stream())

    st.session_state.messages.append({"role": "assistant", "content": full_response})