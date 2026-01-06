"""
Streamlit app for Allama RAG System
Simple UI for testing the chatbot
"""
import sys
import streamlit as st
import os

# Import evidence builder directly
try:
    from src.reasoning.evidence_builder import build_evidence_answer
    from src.core.logging import logger
    _MODEL_IMPORT_ERROR = None
except Exception as _e:
    build_evidence_answer = None
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
    # If model code failed to import, show helpful message
    if _MODEL_IMPORT_ERROR:
        st.session_state.ready = False
        st.session_state.error = (
            "Import failed: " + _MODEL_IMPORT_ERROR
        )
        st.session_state.builder = None
    else:
        st.session_state.ready = True
        st.session_state.builder = build_evidence_answer
        st.session_state.error = None

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
                # Use evidence builder directly (already imported at top)
                answer, references = st.session_state.builder(question)
                
                st.markdown("---")
                st.markdown("### 📖 Jawab")
                st.markdown(answer)
                
                if references:
                    st.markdown("---")
                    st.markdown("### 📺 Video Links (Exact Timestamps)")
                    for i, src in enumerate(references, 1):
                        st.markdown(f"""
                        <div class="source-box">
                        <strong>Video {i}: {src.get('title', 'N/A')}</strong><br>
                        <em>Time: {src.get('time', 'N/A')}</em><br>
                        <a href="{src.get('url', '#')}" target="_blank">🎬 Video dekho →</a>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("⚠️ Is topic par video transcripts mein koi match nahi mila.")
                
                # Show stats
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📺 Videos Found", len(references))
                with col2:
                    st.metric("📝 Answer Length", f"{len(answer)} chars")
                with col3:
                    st.metric("❓ Question Length", f"{len(question)} chars")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
