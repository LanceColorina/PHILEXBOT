# test for upload_qdrant.py
from server.modules.upload_qdrant import upload, query
from qdrant_client import QdrantClient  # Import for type-hinting mock


def test_upload_calls_client_upsert(mocker):
    # Tests that the upload function calls the client's 'upsert' method correctly.
    # 1. ARRANGE
    mock_client = mocker.MagicMock(spec=QdrantClient)

    test_chunks = [{
        "embedding": [0.1, 0.2],
        "text": "test",
        "chunk_id": 1,
        "tokens": 1,
        "page": 1,
        "position_label": "P1"
    }]

    #2. ACT
    upload(test_chunks, mock_client, collection_name="test_docs")

    #3. ASSERT
    call_args = mock_client.upsert.call_args[1]  # [1] gets keyword args
    assert call_args["collection_name"] == "legal_docs"
    assert call_args["points"][0]["payload"]["text"] == "test"

    # print on success
    print(f"\n✅ PASSED: test_upload_calls_client_upsert"
          f"\n   -> Test: client.upsert() was called with the correct data.")


def test_query_calls_client_search(mocker):
    #Tests that the query function calls the client's 'search' method.
    #1. ARRANGE
    mock_client = mocker.MagicMock(spec=QdrantClient)
    mock_client.search.return_value = "fake_search_results"
    query_embedding = [0.5, 0.5]

    #2. ACT
    results = query(mock_client, query_embedding)

    #3. ASSERT
    mock_client.search.assert_called_with(
        collection_name="legal_docs",
        query_vector=query_embedding,
        limit=5
    )
    assert results == "fake_search_results"

    # print on success
    print(f"\n✅ PASSED: test_query_calls_client_search"
          f"\n   -> Test: client.search() was called and returned mock results.")