import streamlit as st
from PyPDF2 import PdfReader
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss faiss

# إعدادات الصفحة
st.set_page_config(page_title="Firdous Local AI", page_icon="🛡️", layout="wide")

# --- تحميل نموذج الـ Embeddings محلياً ---
@st.cache_resource
def load_model():
    # نموذج خفيف وفعال يدعم العربية والإنجليزية محلياً
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# --- CSS مخصص (Glassmorphism & RTL) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; direction: rtl; text-align: right; }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: white;
    }

    /* تأثير الزجاج الشفاف */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    .chat-bubble {
        background: rgba(255, 255, 255, 0.08);
        padding: 20px;
        border-radius: 15px;
        border-right: 5px solid #6366f1;
        margin: 15px 0;
        line-height: 1.6;
    }

    .stButton>button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- منطق استخراج النصوص والبحث الدلالي ---
def process_files(uploaded_files):
    all_chunks = []
    for uploaded_file in uploaded_files:
        text = ""
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        else:
            text = uploaded_file.getvalue().decode("utf-8")
        
        # تقسيم النص إلى فقرات
        chunks = [p.strip() for p in text.split('\n') if len(p.strip()) > 30]
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": uploaded_file.name})
    return all_chunks

def get_semantic_results(query, chunks, top_k=3):
    texts = [c['text'] for c in chunks]
    # تحويل النصوص والمتطلبات إلى Embeddings
    embeddings = model.encode(texts)
    query_embedding = model.encode([query])
    
    # بناء فهرس FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    # البحث عن أقرب المتجهات
    D, I = index.search(np.array(query_embedding).astype('float32'), k=top_k)
    
    results = []
    for idx in I[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    return results

# --- الواجهة الرسومية ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.title("🛡️ فردوس Local AI")
st.write("بحث دلالي محلي بالكامل (بدون إنترنت أو API) يدعم PDF و TXT")
st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("الملفات")
    uploaded_files = st.file_uploader("ارفع ملفاتك هنا", type=["pdf", "txt"], accept_multiple_files=True)
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_files:
    if 'processed_chunks' not in st.session_state or len(uploaded_files) != st.session_state.get('file_count', 0):
        with st.spinner('جاري تحليل الملفات محلياً...'):
            st.session_state.processed_chunks = process_files(uploaded_files)
            st.session_state.file_count = len(uploaded_files)
            st.success("تمت معالجة الملفات!")

    query = st.text_input("ما الذي تبحث عنه في الملفات؟")
    
    if query:
        with st.spinner('جاري البحث الدلالي...'):
            results = get_semantic_results(query, st.session_state.processed_chunks)
            
            st.subheader("أدق الفقرات ذات الصلة بالمعنى:")
            for res in results:
                st.markdown(f"""
                    <div class="chat-bubble">
                        <small style="color: #818cf8;">المصدر: {res['source']}</small><br>
                        {res['text']}
                    </div>
                """, unsafe_allow_html=True)
else:
    st.info("قم برفع ملفاتك من القائمة الجانبية للبدء.")
