import streamlit as st
import json
from openai import OpenAI

# 1. SETUP PAGE CONFIGURATION
st.set_page_config(page_title="Nitra Support", page_icon="💳")

# 2. LOAD FAQ DATA
@st.cache_data
def load_faq():
    try:
        with open('faq_data.json', 'r') as file:
            data = json.load(file)
            # Convert JSON list to a string so the AI can read it
            return json.dumps(data['questions'])
    except FileNotFoundError:
        st.error("FAQ file not found.")
        return ""

faq_text = load_faq()

# 3. INITIALIZE OPENAI CLIENT
# This pulls the key safely from Streamlit Secrets
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("OpenAI API Key is missing. Please add it to Streamlit Secrets.")
    st.stop()

# 4. SYSTEM PROMPT (The Brain Instructions)
# This tells the AI who it is and gives it the knowledge base
system_prompt = f"""
You are a professional, helpful customer support agent for a fintech company called Nitra.
Your goal is to answer user questions accurately using ONLY the context provided below.

Rules:
1. If the answer is found in the CONTEXT, answer politely and concisely.
2. If the user asks a greeting (Hi, Hello), reply naturally.
3. If the answer is NOT in the CONTEXT, say: "I'm sorry, I don't have information on that specific topic. Please contact support@nitra.com for further assistance."
4. Do not make up facts. Stick to the provided data.

CONTEXT:
{faq_text}
"""

# 5. UI LAYOUT
st.title("💳 Nitra Support Assistant")
st.markdown("Welcome! I can help you with cards, expenses, bill pay, and rewards.")

# 6. CHAT HISTORY SETUP
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. CHAT LOGIC
if user_input := st.chat_input("How can I help you today?"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare the conversation history for the AI
        # We start with the System Prompt (Rules + Data)
        messages_payload = [{"role": "system", "content": system_prompt}]
        
        # Add the last few messages from chat history (to keep context)
        for msg in st.session_state.messages[-5:]: 
            messages_payload.append(msg)

        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini", # Smart, fast, and cheap
                messages=messages_payload,
                stream=True,
            )
            
            # Stream the words as they appear (like ChatGPT)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = "I'm having trouble connecting to the server right now."

    # Save AI message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
