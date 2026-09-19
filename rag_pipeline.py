from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split extracted PDF pages into smaller text chunks.

    Each chunk keeps:
    - text
    - source document
    - page number
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = []

    for document in documents:

        text_chunks = splitter.split_text(
            document["text"]
        )

        for chunk in text_chunks:

            chunks.append({
                "text": chunk,
                "source": document["source"],
                "page": document["page"]
            })

    return chunks