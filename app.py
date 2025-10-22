import os
import io
import time
import json
import pickle
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple
from dataclasses import dataclass

# --- Optional: Gemini via google-generativeai ---
USE_GEMINI = False
try:
    import google.generativeai as genai
    if os.getenv("GOOGLE_API_KEY"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        USE_GEMINI = True
except Exception:
    USE_GEMINI = False

# Embedding model (Sentence-Transformers)
from sentence_transformers import SentenceTransformer
# HuggingFace transformers for generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Vector search with FAISS
import faiss

# Dataset loading (WikiRAG-TR) and simple chunking
from datasets import load_dataset

# ------------------- Utility Functions -------------------
def chunk_text(text: str, chunk_size: int = 256, overlap: int = 30) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += max(1, chunk_size - overlap)
    return chunks

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T

def mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int = 4, lambda_mult: float = 0.5) -> List[int]:
    # Maximal Marginal Relevance selection
    selected = []
    candidates = list(range(doc_vecs.shape[0]))
    sim_to_query = (doc_vecs @ query_vec.reshape(-1,1)).ravel() / (
        np.linalg.norm(doc_vecs, axis=1) * (np.linalg.norm(query_vec) + 1e-12) + 1e-12
    )
    while len(selected) < min(k, len(candidates)):
        if not selected:
            idx = int(np.argmax(sim_to_query[candidates]))
            selected.append(candidates.pop(idx))
            continue
        # diversity term
        selected_vecs = doc_vecs[np.array(selected)]
        sim_to_selected = cosine_sim_matrix(doc_vecs[candidates], selected_vecs).max(axis=1)
        mmr_score = lambda_mult*sim_to_query[candidates] - (1-lambda_mult)*sim_to_selected
        next_idx = int(np.argmax(mmr_score))
        selected.append(candidates.pop(next_idx))
    return selected

# ------------------- Cache & Models -------------------
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def load_generator(model_name: str = "google/flan-t5-base"):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tok, model

@st.cache_data(show_spinner=True)
def load_wikirag_subset(max_rows: int = 4000):
    ds = load_dataset("Metin/WikiRAG-TR", split="train")
    df = ds.to_pandas()
    if max_rows is not None:
        df = df.sample(min(max_rows, len(df)), random_state=42).reset_index(drop=True)
    # Build simple documents list with metadata
    docs = []
    for _, r in df.iterrows():
        ctx = r.get("context", "")
        q = r.get("question", "")
        a = r.get("answer", "")
        if isinstance(ctx, str) and ctx.strip():
            for ch in chunk_text(ctx, chunk_size=256, overlap=30):
                docs.append({"content": ch, "meta": {"question": q, "answer": a}})
    return docs

# Building / Loading FAISS index
def build_or_load_faiss(docs: List[Dict], embedder: SentenceTransformer, index_dir: str) -> Tuple[faiss.Index, np.ndarray]:
    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "meta.pkl")
    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return index, meta
    # embed
    texts = [d["content"] for d in docs]
    vectors = embedder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype("float32"))
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(np.array(docs, dtype=object), f)
    return index, np.array(docs, dtype=object)

def retrieve(query: str, top_k: int, embedder: SentenceTransformer, index: faiss.Index, meta_docs: np.ndarray, use_mmr: bool, mmr_k: int, mmr_lambda: float) -> List[Dict]:
    qvec = embedder.encode([query], normalize_embeddings=True)[0].astype("float32")
    D, I = index.search(qvec.reshape(1, -1), max(top_k, mmr_k))
    I = I[0].tolist()
    selected_indices = I[:top_k]
    if use_mmr:
        # Need dense vectors of candidates for MMR
        xb = index.reconstruct_n(0, index.ntotal)  # retrieve all; for FlatIP it's OK
        cand_vecs = xb[I, :]
        mmr_sel_local = mmr(qvec, cand_vecs, k=mmr_k, lambda_mult=mmr_lambda)
        selected_indices = [I[i] for i in mmr_sel_local]
    results = [meta_docs[i].item() if hasattr(meta_docs[i], "item") else meta_docs[i] for i in selected_indices]
    return results

# ------------------- Generation -------------------
SYSTEM_HINT = ("Yalnızca verilen bağlamdan yararlanarak kısa ve net bir yanıt ver. "
               "Bağlam yeterli değilse 'Belgelerde bu konuyla ilgili yeterli bilgi yok.' de. "
               "Uydurma bilgi ekleme.")

def generate_answer(question: str, passages: List[str], answer_len: str, tok, model) -> str:
    max_new_tokens = {"kısa": 96, "orta": 196, "uzun": 256}.get(answer_len, 128)
    context = "\n\n".join(passages[:4])
    prompt = f"{SYSTEM_HINT}\n\nSoru: {question}\n\nBağlam:\n{context}\n\nYanıt:"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(outputs[0], skip_special_tokens=True)
    return text.strip()

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="AtlasRAG: Türkçe Bilgi Asistanı", page_icon="🧠")
st.title("🧠 AtlasRAG: Türkçe Bilgi Asistanı")
st.caption("AtlasRAG, WikiRAG‑TR veri seti üzerinde RAG ile arama yapar ve kaynaklı yanıt üretir. (Kendi verinizi eklemek için `data/` klasörünü kullanabilirsiniz.)")

with st.sidebar:
    st.header("⚙️ Ayarlar")
    top_k = st.slider("Top‑K (kand.)", 3, 20, 10)
    use_mmr = st.checkbox("MMR re‑ranking", True)
    mmr_k = st.slider("MMR k", 2, 10, 4)
    mmr_lambda = st.slider("MMR λ", 0.1, 0.9, 0.5)
    answer_len = st.selectbox("Yanıt uzunluğu", ["kısa", "orta", "uzun"], index=1)
    subset_rows = st.selectbox("Veri boyutu", ["4k (hızlı)", "Tümü (yavaş)"], index=0)
    st.divider()
    if USE_GEMINI:
        st.success("Gemini etkin (GOOGLE_API_KEY bulundu).")
    else:
        st.info("Gemini kapalı. Yerel flan‑t5 kullanılıyor.")

# Load models & data
with st.spinner("Modeller yükleniyor..."):
    embedder = load_embedder()
    tok, model = load_generator()

max_rows = 4000 if subset_rows.startswith("4k") else None
with st.spinner("Veri yükleniyor ve indeks hazırlanıyor... (ilk çalıştırma uzun sürebilir)"):
    docs = load_wikirag_subset(max_rows=max_rows)
    index, meta_docs = build_or_load_faiss(docs, embedder, index_dir="index")

st.success(f"Belgeler: {len(meta_docs)} | Embed: paraphrase-multilingual-MiniLM-L12-v2 | Index: FAISS (IP)")

# Chat area
if "history" not in st.session_state:
    st.session_state["history"] = []

for m in st.session_state["history"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Sorunuzu yazın (ör. 'Türkiye'de siber güvenlik tezleri...')")
if q:
    st.session_state["history"].append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.spinner("Belgeler taranıyor ve yanıt hazırlanıyor..."):
        results = retrieve(q, top_k=top_k, embedder=embedder, index=index, meta_docs=meta_docs,
                           use_mmr=use_mmr, mmr_k=mmr_k, mmr_lambda=mmr_lambda)
        passages = [r["content"] for r in results]
        if len(passages) == 0:
            response = "Belgelerde bu konuyla ilgili yeterli bilgi yok."
        else:
            if USE_GEMINI:
                # Short Gemini generation (if available)
                ctx = "\n\n".join(passages[:4])
                prompt = f"{SYSTEM_HINT}\n\nSoru: {q}\n\nBağlam:\n{ctx}\n\nYanıt:"
                response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt).text
            else:
                response = generate_answer(q, passages, answer_len, tok, model)

    # Render assistant message with sources
    src_lines = []
    for r in results:
        meta = r.get("meta", {})
        t = meta.get("question", "")[:60]
        src_lines.append(f"- **Kaynak parça:** {t}...")
    final_msg = response + ("\n\n**Kaynaklar:**\n" + "\n".join(src_lines) if src_lines else "")
    st.session_state["history"].append({"role": "assistant", "content": final_msg})

    with st.chat_message("assistant"):
        st.markdown(final_msg)

with st.expander("ℹ️ Proje Bilgisi"):
    st.markdown("""
**RAG Mimarisi:** Sentence‑Transformers (çokdilli) ile embedding → FAISS (inner‑product) arama → 
MMR re‑ranking → Flan‑T5 (veya opsiyonel Gemini) ile üretim.  
**Gizlilik:** Uygunsuz içerikler filtrelenmez; teslim öncesi basit denetim ekleyiniz.  
**Performans:** İlk indeksleme cache edilir; sonraki açılışlar hızlıdır.
""")