import os
from qdrant_client import QdrantClient
from uuid import uuid4
def upload(chunks: list[dict], client: QdrantClient, collection_name: str = "legal_docs") -> bool:
    for c in chunks:
        client.upsert(
            collection_name="legal_docs",
            points=[{
                "id": str(uuid4()),
                "vector": c["embedding"],
                "payload": {
                    "text": c['text'],
                    "chunk_id": c['chunk_id'],
                    "token_count": c['tokens'],
                    "page": c['page'],
                    "chunk_method": "Semantic",
                    "position_label": c['position_label']
                }
            }]
        )
    return True

def query(client:QdrantClient, query_embedding: str):
    return client.search(
                collection_name="legal_docs",
                query_vector=query_embedding,
                limit=5
            )

def main():
    """
    Test function for uploading and querying dummy embeddings in Qdrant.
    """
    print("🔍 Testing Qdrant upload and query module...\n")

    # Load credentials from environment (recommended)
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    if not QDRANT_URL or not QDRANT_API_KEY:
        print("⚠️ Missing credentials. Please set environment variables:")
        print("   set QDRANT_URL=<your_url>")
        print("   set QDRANT_API_KEY=<your_api_key>")
        return

    # Connect to Qdrant Cloud
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Dummy test data
    chunks = [
        {
            "embedding": [0.12, 0.47, 0.33, 0.89, 0.51],  # Replace with actual embeddings
            "text": "This is a test legal clause about contract obligations.",
            "chunk_id": 1,
            "tokens": 15,
            "page": 1,
            "position_label": "W1"
        }
    ]

    # Upload
    print("⬆️ Uploading test chunk to Qdrant...")
    success = upload(chunks, client)
    print("✅ Upload success:", success)

    # Query using the same embedding
    print("\n🔎 Running similarity search...")
    results = query(client, chunks[0]["embedding"])

    for i, r in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Score: {r.score:.4f}")
        print(f"  Text: {r.payload.get('text', '')[:150]}...")


if __name__ == "__main__":
    main()