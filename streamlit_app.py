"""
Streamlit app for Allama RAG System
Simple UI for testing the chatbot
"""
import sys
import streamlit as st
import os

# Import model code lazily and handle import errors so Streamlit UI doesn't crash
try:
    from src.chat.chat_model import ChatModel
    from src.chat.model_loader import load_model
    from src.core.logging import logger
    _MODEL_IMPORT_ERROR = None
except Exception as _e:
    ChatModel = None
    load_model = None
    logger = None
    _MODEL_IMPORT_ERROR = str(_e)

# Set page config
st.set_page_config(page_title="Allama RAG", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stTextArea { max-width: 100%; }
    .response-box { 
        background-color: #f0f2f6; 
        padding: 1.5rem; 
        border-radius: 10px; 
        border-left: 5px solid #1f77b4;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #0d9488;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_model' not in st.session_state:
    # If model code failed to import (e.g. Keras/transformers dependency issue),
    # avoid raising on import and show helpful message in UI instead.
    if _MODEL_IMPORT_ERROR:
        st.session_state.ready = False
        st.session_state.error = (
            "Model import failed: " + _MODEL_IMPORT_ERROR + ".\n"
            "If you see a Keras 3 compatibility error, run: `pip install tf-keras` "
            "or install compatible transformer/sentence-transformers versions."
        )
        st.session_state.chat_model = None
    else:
        with st.spinner("⏳ Model load ho raha hai..."):
            try:
                llm = load_model()
                st.session_state.chat_model = ChatModel(llm=llm)
                st.session_state.ready = True
            except Exception as e:
                st.session_state.ready = False
                st.session_state.error = str(e)

# Header
st.title("🎓 Allama RAG System")
st.markdown("**AI-powered Q&A from Islamic video lectures**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("### Model Status")
    if st.session_state.ready:
        st.success("✅ Model Ready")
    else:
        st.error(f"❌ Error: {st.session_state.error}")
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    **Allama** ek AI chatbot hai jo Islamic videos ke transcripts se seekh ke
    aapke sawalon ka jawab deta hai.
    
    **Features:**
    - 🌍 Multi-language support
    - 🎥 YouTube timestamps ke sath links
    - 📚 4500+ video segments indexed
    - ⚡ Fast semantic search
    """)
    
    st.markdown("---")
    st.markdown("### Tips")
    st.write("""
    - Urdu/Arabic mein sawal poocho
    - Hindu mein poocho  
    - English mein poocho
    - Specific topic poochna better hai
    """)

# Main content
if not st.session_state.ready:
    st.error("❌ Model load nahi hua. Kuch error hai.")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🤔 Aapka Sawal")
    question = st.text_area(
        "Yahan sawal likho...",
        placeholder="e.g., Imaan kya hota hai? / What is Imaan? / نماز کیا ہے",
        height=100,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### ⚡ Quick Examples")
    examples = [
        "What is Imaan?",
        "نماز کیا ہے؟",
        "Quran kaun sa kitaab hai?",
        "Allah ke 99 naam kya hain?"
    ]
    for example in examples:
        if st.button(f"📌 {example}", use_container_width=True):
            question = example

# Process question
if st.button("🔍 Jawab Lao", use_container_width=True, type="primary"):
    if not question.strip():
        st.warning("⚠️ Pehle sawal likho!")
    else:
        with st.spinner("🤖 Soch raha hoon..."):
            try:
                result = st.session_state.chat_model.answer(question)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                
                st.markdown("---")
                st.markdown("### 📖 Jawab")
                st.markdown(answer)
                
                if sources:
                    st.markdown("---")
                    st.markdown("### 📺 Video Links (Timestamps ke Sath)")
                    for i, src in enumerate(sources, 1):
                        # Extract timestamp from URL for display
                        timestamp = ""
                        if "&t=" in src:
                            t_val = src.split("&t=")[-1].replace("s", "")
                            try:
                                secs = int(t_val)
                                mins = secs // 60
                                secs = secs % 60
                                timestamp = f" ({mins}:{secs:02d})"
                            except:
                                pass
                        
                        st.markdown(f"""
                        <div class="source-box">
                        <strong>Video {i}{timestamp}</strong><br>
                        <a href="{src}" target="_blank">🎬 Click karo video dekhnay ke liye →</a>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("⚠️ Is topic par video transcripts mein koi match nahi mila.")
                
                # Show stats
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Videos Found", len(sources))
                with col2:
                    st.metric("Answer Chars", f"{len(answer)}")
                with col3:
                    st.metric("Query Chars", f"{len(question)}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                if logger:
                    logger.error(f"Chat error: {e}", exc_info=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
    <p>Made with ❤️ | Allama RAG System v1.0 | 2026</p>
    <p>📧 For issues, visit: <a href="#">GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)
