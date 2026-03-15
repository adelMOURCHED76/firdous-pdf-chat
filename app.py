import streamlit as st
import pdfplumber
import re
import io
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# إعدادات الصفحة
st.set_page_config(page_title="Firdous Local AI", page_icon="🛡️", layout="wide")

# --- تحميل نموذج الـ Embeddings محلياً ---
@st.cache_resource
def load_model():
    # نموذج خفيف وفعال يدعم العربية والإنجليزية محلياً
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# محاولة تحميل النموذج مع معالجة الأخطاء
try:
    model = load_model()
except Exception as e:
    st.error(f"خطأ في تحميل النموذج: {e}")
    st.stop()

# --- CSS مخصص (Glassmorphism & RTL) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    
    .main {
        direction: rtl;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: white;
        font-family: 'Cairo', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.9) !important;
        backdrop-filter: blur(10px);
        border-left: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: #cbd5e1;
        text-align: right;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        color: white;
    }

    .chat-bubble {
        background: rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        border-radius: 15px;
        border-right: 5px solid #6366f1;
        margin: 1rem 0;
        line-height: 1.6;
        color: #f1f5f9;
        text-align: right;
    }

    .stButton > button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        font-weight: 700 !important;
        width: 100%;
        transition: all 0.3s ease !important;
        font-family: 'Cairo', sans-serif !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(79, 70, 229, 0.3) !important;
        opacity: 0.9;
    }

    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.07) !important;
        color: white !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 12px !important;
        text-align: right !important;
        direction: rtl !important;
    }
    
    .stTextInput label {
        color: #94a3b8 !important;
        text-align: right !important;
        width: 100%;
    }

    .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        text-align: right;
        direction: rtl;
    }

    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- منطق استخراج النصوص والبحث الدلالي ---
def process_files(uploaded_files):
    all_chunks = []
    for uploaded_file in uploaded_files:
        text = ""
        try:
            if uploaded_file.type == "application/pdf":
                # استخدام pdfplumber بدلاً من PyPDF2 لدقة أعلى في العربية
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
            else:
                text = uploaded_file.getvalue().decode("utf-8")
        except Exception as e:
            st.error(f"خطأ في قراءة الملف {uploaded_file.name}: {e}")
            continue
        
        # تنظيف النص من الرموز الغريبة الناتجة عن أخطاء التشفير
        text = re.sub(r'[^\w\s\.\!\؟\،\:\-\(\)]', '', text)
        
        # تقسيم النص إلى فقرات ذات معنى
        chunks = [p.strip() for p in text.split('\n') if len(p.strip()) > 30]
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": uploaded_file.name})
    return all_chunks

def get_semantic_results(query, chunks, top_k=3):
    if not chunks:
        return []
    
    texts = [c['text'] for c in chunks]
    # تحويل النصوص والمتطلبات إلى Embeddings
    embeddings = model.encode(texts)
    query_embedding = model.encode([query])
    
    # بناء فهرس FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    # البحث عن أقرب المتجهات
    D, I = index.search(np.array(query_embedding).astype('float32'), k=min(top_k, len(chunks)))
    
    results = []
    for idx in I[0]:
        if idx != -1 and idx < len(chunks):
            results.append(chunks[idx])
    return results

# --- الواجهة الرسومية ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.title("🛡️ فردوس Local AI")
st.markdown("### بحث دلالي محلي بالكامل (بدون إنترنت أو API) يدعم PDF و TXT")
st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("📂 إدارة الملفات")
    uploaded_files = st.file_uploader("ارفع ملفاتك هنا", type=["pdf", "txt"], accept_multiple_files=True)
    st.markdown('---')
    st.info("يتم تحليل الملفات محلياً على جهازك لضمان الخصوصية التامة.")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_files:
    # استخدام session_state لتخزين البيانات المعالجة
    if 'processed_chunks' not in st.session_state or len(uploaded_files) != st.session_state.get('file_count', 0):
        with st.spinner('جاري تحليل الملفات وبناء قاعدة البيانات المتجهة...'):
            st.session_state.processed_chunks = process_files(uploaded_files)
            st.session_state.file_count = len(uploaded_files)
            if st.session_state.processed_chunks:
                st.success(f"تمت معالجة {len(uploaded_files)} ملفات بنجاح!")
            else:
                st.warning("لم يتم العثور على نصوص كافية في الملفات المرفوعة.")

    if st.session_state.get('processed_chunks'):
        query = st.text_input("🔍 ما الذي تبحث عنه في الملفات؟ (سأفهم المعنى حتى لو اختلفت الكلمات)")
        
        if query:
            with st.spinner('جاري البحث عن أدق الفقرات...'):
                results = get_semantic_results(query, st.session_state.processed_chunks)
                
                if results:
                    st.markdown("### 🎯 أدق الفقرات ذات الصلة:")
                    for res in results:
                        st.markdown(f"""
                            <div class="chat-bubble">
                                <small style="color: #818cf8; font-weight: bold;">📍 المصدر: {res['source']}</small><br>
                                {res['text']}
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("لم يتم العثور على نتائج مطابقة للمعنى.")
else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; opacity: 0.7;">
        <h3>👋 أهلاً بك في فردوس AI</h3>
        <p>يرجى رفع ملفات PDF أو TXT من القائمة الجانبية للبدء في البحث الذكي.</p>
    </div>
    """, unsafe_allow_html=True)
