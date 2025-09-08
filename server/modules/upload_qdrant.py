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

def query(client:QdrantClient):
    return client.search(
                collection_name="legal_docs",
                query_vector=query_embedding,
                limit=5
            )