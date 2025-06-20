import os
import re
import fitz  # PyMuPDF
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import disk_offload
import torch
import numpy as np
from uuid import uuid4

# RUN docker first: docker run -p 6333:6333 qdrant/qdrant 
# RUN : .env\Scripts\Activate
# RUN : streamlit run qdrantSetup.py

# -----------------------------
# Load LLaMA 3 (8B)
# -----------------------------
@st.cache_resource
def load_chat_pipeline():
    # model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    model_id = "openchat/openchat-3.5-1210"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0  # force GPU (CUDA:0) instead of default auto-inference
    )


chat_pipeline = load_chat_pipeline()

# Load LEGAL-BERT
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-uncased-contracts")
model = AutoModel.from_pretrained("nlpaueb/bert-base-uncased-contracts")

# -----------------------------
# Embedding Function
# -----------------------------
def get_legal_bert_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().numpy()

# -----------------------------
# Sanitize Sensitive Data
# -----------------------------
def sanitize(text):
    patterns = {
        'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\w{2,}\b',
        'PHONE': r'\b\d{10,}\b',
        'ZIP': r'\b\d{5}\b',
        'ADDRESS': r'\b\d{1,3} [A-Za-z ]+,? [A-Za-z]{2,},? \d{5}\b',
        'NAME': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
    }

    mask_map = {}
    reverse_map = {}
    counters = {key: 1 for key in patterns.keys()}

    def replacer(match, label):
        original = match.group(0)
        if original in reverse_map:
            return f"[{reverse_map[original]}]"
        else:
            mask_key = f"{label}_{counters[label]}"
            mask_map[mask_key] = original
            reverse_map[original] = mask_key
            counters[label] += 1
            return f"[{mask_key}]"

    for label, pattern in patterns.items():
        text = re.sub(pattern, lambda m: replacer(m, label), text)

    return text, mask_map

# -----------------------------
# Sentence and Token-Based Chunking
# -----------------------------
def split_sentences(text: str):
    return re.split(r'(?<=[\.!?])\s+', text)

def chunk_text_tokenwise(text: str, tokenizer, max_tokens=300, overlap_tokens=50):
    sentences = split_sentences(text)
    chunks = []
    current_sents = []
    current_ids = []
    chunk_id = 0

    def flush_chunk(cid):
        if current_ids:
            chunk_text = ' '.join(current_sents)
            chunks.append({
                'chunk_id': cid,
                'text': chunk_text,
                'tokens': len(current_ids)
            })

    for sent in sentences:
        sent_ids = tokenizer.encode(sent, add_special_tokens=False)
        if len(current_ids) + len(sent_ids) > max_tokens:
            flush_chunk(chunk_id)
            chunk_id += 1
            overlap_ids = current_ids[-overlap_tokens:] if len(current_ids) > overlap_tokens else current_ids
            overlap_text = tokenizer.decode(overlap_ids, skip_special_tokens=True)
            current_sents = [overlap_text, sent]
            current_ids = tokenizer.encode(overlap_text, add_special_tokens=False) + sent_ids
        else:
            current_sents.append(sent)
            current_ids.extend(sent_ids)

    flush_chunk(chunk_id)
    return chunks

# -----------------------------
# Initialize Qdrant
# -----------------------------
client = QdrantClient(host="localhost", port=6333)

client.recreate_collection(
    collection_name="legal_docs",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

# -----------------------------
# Answer Generation
# -----------------------------
def generate_answer(query: str, top_chunks: list[str]) -> str:
    context = "\n\n".join(top_chunks)
    prompt = f"""You are a legal assistant AI. Use the context below to answer the question.

Context:
{context}

Question: {query}
Answer:"""

    response = chat_pipeline(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
    return response[0]["generated_text"].split("Answer:")[-1].strip()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Legal PDF Embedder (Qdrant + LEGAL-BERT)")
uploaded_file = st.file_uploader("Upload a legal PDF file", type=["pdf"])

if uploaded_file is not None:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        full_text = ""
        for page in doc:
            full_text += page.get_text()

    st.subheader("📄 Raw Extracted Text")
    st.write(full_text[:1000] + "...")

    # Sanitize and chunk
    sanitized, mask_map = sanitize(full_text)
    chunks = chunk_text_tokenwise(sanitized, tokenizer)

    st.subheader("🧠 Mask Mapping")
    st.json(mask_map)

    st.subheader("🔐 Sanitized & Chunked")
    st.write(f"Total chunks: {len(chunks)}")

    st.markdown("### 🧩 Example Chunks")
    for c in chunks[:2]:
        st.code(c['text'], language='text')

    st.info("Embedding and uploading to Qdrant...")

    for c in chunks:
        embedding = get_legal_bert_embedding(c['text'])
        client.upsert(
            collection_name="legal_docs",
            points=[{
                "id": str(uuid4()),
                "vector": embedding.tolist(),
                "payload": {
                    "text": c['text'],
                    "chunk_id": c['chunk_id'],
                    "token_count": c['tokens']
                }
            }]
        )

    st.success("✅ Embeddings uploaded to Qdrant!")

    st.markdown("---")
    st.header("💬 Ask a Question")

    query = st.text_input("Enter your legal question")

    if query:
        query_embedding = get_legal_bert_embedding(query)
        search_result = client.search(
            collection_name="legal_docs",
            query_vector=query_embedding.tolist(),
            limit=5
        )
        retrieved_chunks = [hit.payload['text'] for hit in search_result]

        st.subheader("📚 Retrieved Context Chunks")
        for chunk in retrieved_chunks:
            st.code(chunk, language='text')

        st.subheader("🤖 Answer")
        answer = generate_answer(query, retrieved_chunks)
        st.write(answer)



