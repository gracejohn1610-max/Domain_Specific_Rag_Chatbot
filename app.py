import os
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from document_loader import extract_text_from_pdf
from rag_pipeline import split_documents
from vector_store import create_vector_store, search_vector_store
from prompt import build_prompt

load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# Page configuration
st.set_page_config(
    page_title="Domain-Specific RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# Main title
st.title("📚 Domain-Specific RAG Chatbot")

st.write(
    "Upload PDF documents and ask questions based on their content."
)


# Sidebar
st.sidebar.header("📄 Upload Documents")


uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# Display uploaded files
if uploaded_files:

    st.sidebar.success(
        f"{len(uploaded_files)} PDF file(s) uploaded."
    )

    st.sidebar.subheader("Uploaded Documents")

    for file in uploaded_files:
        st.sidebar.write(f"📄 {file.name}")

else:

    st.sidebar.info(
        "Please upload at least one PDF."
    )


st.divider()


# Document processing section
st.subheader("📖 Document Processing")


if uploaded_files:

    if st.button("🔄 Process Documents"):

        all_documents = []

        for uploaded_file in uploaded_files:

            documents = extract_text_from_pdf(
                uploaded_file
            )

            all_documents.extend(documents)


        # Store extracted documents
        st.session_state["documents"] = all_documents
        # Split documents into chunks
        chunks = split_documents(all_documents)

        st.session_state["chunks"] = chunks
        index, stored_chunks = create_vector_store(chunks)

        st.session_state["vector_index"] = index
        st.session_state["stored_chunks"] = stored_chunks
        st.success(
    f"Successfully extracted text from "
    f"{len(all_documents)} page(s) "
    f"and created {len(chunks)} text chunks."
)

        st.success(
            f"Successfully extracted text from "
            f"{len(all_documents)} page(s)."
        )


        # Show extraction information
        st.write("### Extracted Pages")

        for document in all_documents:

            st.write(
                f"📄 **{document['source']}** "
                f"- Page {document['page']}"
            )


else:

    st.info(
        "Upload PDF documents from the sidebar to begin."
    )

    st.divider()

st.subheader("💬 Ask a Question")

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state["chat_history"] = []
    st.rerun()


# Display previous conversation
for message in st.session_state["chat_history"]:

    if message["role"] == "user":

        st.write("### 👤 You")
        st.write(message["content"])

    else:

        st.write("### 🤖 Assistant")
        st.write(message["content"])

        if "sources" in message:

            st.write("### 📚 Sources")

            for source in message["sources"]:
                st.write(
                    f"- **{source['source']}** "
                    f"— Page {source['page']}"
                )


# Ask a new question
if (
    "vector_index" in st.session_state
    and st.session_state["vector_index"] is not None
):

    question = st.chat_input(
        "Ask a question about your uploaded PDF..."
    )

    if question:

        # Show user's question immediately
        st.session_state["chat_history"].append({
            "role": "user",
            "content": question
        })

        # Retrieve relevant document chunks
        results = search_vector_store(
            st.session_state["vector_index"],
            st.session_state["stored_chunks"],
            question,
            top_k=5
        )

        # Build context
        context_parts = []

        for result in results:

            context_parts.append(
                f"Source: {result['source']}\n"
                f"Page: {result['page']}\n"
                f"Content:\n{result['text']}"
            )

        context = "\n\n".join(context_parts)

        # Build grounded prompt
        prompt = build_prompt(
            question,
            context
        )

        # Get API key
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:

            st.error(
                "GROQ_API_KEY was not found. "
                "Please check your .env file."
            )

        else:

            # Initialize Groq
            llm = ChatGroq(
                model="openai/gpt-oss-20b",
                temperature=0,
                api_key=api_key
            )

                        # Generate answer
            response = llm.invoke(prompt)

            answer = response.content

            # Store sources only when an answer was found
            sources = []

            if "I cannot find the answer in the uploaded documents." not in answer:

                for result in results:

                    source_info = {
                        "source": result["source"],
                        "page": result["page"]
                    }

                    if source_info not in sources:
                        sources.append(source_info)

            # Store assistant response
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })            

            # Refresh page to display the conversation
            st.rerun()

else:

    st.info(
        "Please upload and process a PDF "
        "before asking a question."
    )