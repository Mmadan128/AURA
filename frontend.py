import streamlit as st
import requests
import json
import base64
import time
import uuid
import os

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Helper functions
def autoplay_audio(audio_bytes: bytes):
    """Auto-play audio in the browser"""
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    md = f"""
    <audio autoplay="true">
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

def check_backend_health():
    """Check if backend is healthy and initialized"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "initialized": False, "error": f"Backend returned status {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "initialized": False, "error": f"Cannot connect to backend at {BACKEND_URL}. Make sure the backend is running."}
    except requests.exceptions.Timeout:
        return {"status": "error", "initialized": False, "error": "Backend request timed out"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "initialized": False, "error": f"Connection error: {str(e)}"}

def get_init_status():
    """Get detailed initialization status"""
    try:
        response = requests.get(f"{BACKEND_URL}/init-status", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"initialized": False, "error": "Failed to get status", "build_progress": {}}
    except requests.exceptions.RequestException:
        return {"initialized": False, "error": "Connection error", "build_progress": {}}

def send_chat_message(message: str, session_id: str):
    """Send chat message to backend"""
    try:
        payload = {
            "message": message,
            "session_id": session_id
        }
        response = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Backend error: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}

def stream_chat_response(message: str, session_id: str):
    """Stream chat response from backend"""
    try:
        payload = {
            "message": message,
            "session_id": session_id
        }
        response = requests.post(
            f"{BACKEND_URL}/chat/stream", 
            json=payload, 
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        try:
                            data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                            if 'chunk' in data:
                                yield data['chunk']
                            elif data.get('done'):
                                break
                        except json.JSONDecodeError:
                            continue
        else:
            yield f"Error: Backend returned status {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        yield f"Connection error: {str(e)}"

def generate_audio(text: str, lang_code: str = 'en'):
    """Generate audio from text"""
    try:
        payload = {
            "text": text,
            "lang_code": lang_code
        }
        response = requests.post(f"{BACKEND_URL}/generate-audio", json=payload, timeout=15)
        if response.status_code == 200:
            return response.content
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def clear_session_memory(session_id: str):
    """Clear session memory on backend"""
    try:
        response = requests.delete(f"{BACKEND_URL}/memory/{session_id}", timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_sources(session_id: str, query: str):
    """Get retrieved sources for a query"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/sources/{session_id}",
            params={"query": query},
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get("sources", [])
        else:
            return []
    except requests.exceptions.RequestException:
        return []

# Streamlit UI Configuration
st.set_page_config(
    page_title="Project AURA", 
    page_icon="🤖", 
    layout="centered"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("🤖 Project AURA")
    st.subheader("The Autonomous, Multimodal Campus Assistant")
    
    # Backend status
    health_status = check_backend_health()
    
    if health_status["initialized"]:
        st.success("✅ Backend Ready")
    else:
        st.error("❌ Backend Not Ready")
        st.info(f"Error: {health_status.get('error', 'Unknown error')}")
        
        # Show initialization progress if building index
        init_status = get_init_status()
        if init_status.get("build_progress", {}).get("total", 0) > 0:
            progress = init_status["build_progress"]
            if not progress.get("completed", False):
                st.progress(
                    progress["current"] / progress["total"],
                    text=f"Building index... {progress['current']}/{progress['total']} - {progress.get('file', '')}"
                )
    
    st.info("AURA can now remember your conversation. Ask follow-up questions!")
    
    # Clear conversation button
    if st.button("Clear Conversation History"):
        if clear_session_memory(st.session_state.session_id):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.success("History and memory cleared!")
            st.rerun()
        else:
            st.error("Failed to clear memory")
    
    # Rebuild index button
    if st.button("Rebuild Document Index"):
        try:
            response = requests.post(f"{BACKEND_URL}/rebuild-index", timeout=5)
            if response.status_code == 200:
                st.success("Index rebuild started!")
                st.info("Check the status above for progress")
            else:
                st.error("Failed to start rebuild")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")
    
    st.markdown("---")
    st.markdown("Powered by LangChain, Google Gemini, FastAPI, and Streamlit.")

# Main UI
st.title("Project AURA Chat")
st.caption("Now with conversational memory and streaming responses via FastAPI backend.")

# Check if backend is ready
if not health_status["initialized"]:
    st.error("🚫 Backend is not ready. Please wait for initialization to complete.")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        source_name = source.get("source", "N/A")
                        if source_name != "N/A":
                            source_name = source_name.split("/")[-1]  # Get filename only
                        st.write(f"**Source:** `{source_name}`")
                        content = source.get("content", "")
                        st.write(f"**Content:**\n```\n{content[:150]}...\n```")

# Voice input using Streamlit's built-in audio input
st.subheader("🎤 Voice Input")
audio_bytes = st.audio_input("Record a voice message")
if audio_bytes:
    st.info("Voice input received! For this demo, please type your query in the text box and press Enter.")
    # You could add speech-to-text processing here in the future

# Chat input
if prompt := st.chat_input("Ask AURA about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.status("AURA is processing...", expanded=True) as status:
            status.write("🧠 Detecting language...")
            status.write("📚 Searching documents...")
            status.write("✍️ Generating streaming response...")
            
            # Stream the response
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Use streaming response
                for chunk in stream_chat_response(prompt, st.session_state.session_id):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                # Get sources after response is complete
                status.write("📖 Retrieving sources...")
                sources = get_sources(st.session_state.session_id, prompt)
                
                # Generate audio
                status.write("🗣️ Generating audio...")
                audio_bytes = generate_audio(full_response)
                if audio_bytes:
                    autoplay_audio(audio_bytes)
                    status.write("🔊 Audio ready!")
                else:
                    status.write("⚠️ Audio generation failed")
                
                status.update(label="Response complete!", state="complete", expanded=False)
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": sources
                })
                
            except Exception as e:
                status.update(label=f"Error: {e}", state="error")
                full_response = f"I encountered an error: {str(e)}"
                response_placeholder.markdown(full_response)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": []
                })

# Footer
st.markdown("---")
st.markdown("**Session ID:** " + st.session_state.session_id[:8] + "...")

if __name__ == "__main__":
    # Note: This won't run when imported as a module
    pass