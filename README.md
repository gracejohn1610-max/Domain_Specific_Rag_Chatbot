# Domain-Specific RAG Chatbot for PDF Question Answering

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content.

The system extracts text from uploaded PDFs, divides the text into smaller chunks, converts the chunks into vector embeddings, stores them using FAISS, retrieves the most relevant information for a user's question, and generates an answer using a Groq-hosted Large Language Model.

---

## 🌐 Live Demo

### Test the deployed application

👉 **[Test the RAG Chatbot](https://domainspecific-rag-chatbot.streamlit.app/)**

