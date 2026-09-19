import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def create_vector_store(chunks):
    """
    Create embeddings for document chunks
    and store them in a FAISS index.
    """

    if not chunks:
        return None, []

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, chunks


def search_vector_store(index, chunks, query, top_k=5):
    """
    Find the most relevant document chunks for a question.
    """

    if index is None or not chunks:
        return []

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for distance, index_number in zip(
        distances[0],
        indices[0]
    ):

        if index_number >= 0:

            result = chunks[index_number].copy()

            result["distance"] = float(distance)

            results.append(result)

    return results