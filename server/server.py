# server.py
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from server.modules.pii_sanitizer import PIISanitizer
from server.modules.pdf_text_extractor import extract_text_from_pdf  # <-- import module
from server.modules.pdf_chunker import chunk_pdf_with_semantic       # new chunker module
from server.modules.embed_text import embed_texts
from server.modules.upload_qdrant import upload, query

# from transformers import AutoTokenizer, AutoModel
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:3001"])

sessions = {}
# Example: load tokenizer + model once (avoid reloading every request)
# tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# embedding_model = HuggingFaceEmbeddings(model_name="roberta-base")

client = QdrantClient(
    url="https://4a6f24fc-8fce-4921-bc8d-c3f533294e90.europe-west3-0.gcp.cloud.qdrant.io",  
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.bMlOchp7JkGWmzEcs5HC7h7em8k1U3c_MBuFUF1f_7M",  
)

client.recreate_collection(
    collection_name="legal_docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

sanitizer = PIISanitizer()  # initialize sanitizer once

@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to the Flask server!",
        "status": "running"
    })

@app.route("/session", methods=["POST"])
def create_session():
    session_id = str(uuid4())
    sessions[session_id] = {"chat": [], "pdf": None}

    resp = make_response(jsonify({"status": "ok"}))
    resp.set_cookie("session_id", session_id, httponly=True, samesite="Lax")
    return resp

@app.route("/upload", methods=["POST"])
def upload_pdf():
    session_id = request.cookies.get("session_id")
    print("request.files:", request.files)
    print("request.form:", request.form)
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    pdf_bytes = file.read()

    # Extract text
    page_texts = extract_text_from_pdf(pdf_bytes)
    print("text extracted from PDF")
    # Sanitize
    
    sanitized_pages = []
    for page in page_texts:
        masked_text, mask_map = sanitizer.sanitize(page["text"])
        sanitized_pages.append({
            "page": page["page"],
            "text": masked_text,   # store sanitized text
            "mask_map": mask_map   # keep mapping for restore
        })

    print("text sanitized")

    chunks = chunk_pdf_with_semantic(sanitized_pages)  # embedding_model

    print("text chunked")

    chunks = embed_texts(chunks, model)  # embedding_model

    print("text embedded")

    upload(chunks, client)

    print("uploaded to Qdrant")
    # chunks = chunk_pdf_with_semantic(page_texts, embedding_model)
    sessions[session_id]["pdf"] = {
        "pages": page_texts,
        # "chunks": chunks
    }

    return jsonify({
        "status": "ok",
        "pages": len(page_texts),
        "chunks": [chunk["text"] for chunk in chunks]
    })

@app.route("/chat", methods=["POST"])
def chat():
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 400

    data = request.json
    message = data.get("message")
    # sanitize first the user input

    sanitized_message, _ = sanitizer.sanitize(message)
    print("sanitized user message:", sanitized_message)
    # embed
    embeded_message = embed_texts([{"text": sanitized_message}], model)[0]["embedding"]
    print("embedded user message")
    # query
    results = query(client, embeded_message)
    print("queried Qdrant:", results)

    # generate response (mocked here)

    sessions[session_id]["chat"].append({"role": "user", "content": message})

    # Use PDF context if available
    pdf_context = sessions[session_id].get("pdf")
    context_info = f" (PDF loaded with {len(pdf_context)} pages)" if pdf_context else ""

    reply = f"Bot: You said '{message}'{context_info}"
    sessions[session_id]["chat"].append({"role": "bot", "content": reply})

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True, port=3000)