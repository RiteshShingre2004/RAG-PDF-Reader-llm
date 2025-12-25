# 📄 RAG-based PDF Reader using LLMs

A backend-focused **Retrieval-Augmented Generation (RAG)** system that enables users to query PDF documents and receive context-aware answers using Large Language Models.

This project demonstrates an **industry-grade RAG pipeline** including document ingestion, chunking, vector storage, retrieval, and answer generation — without any UI layer.



 Problem Statement

Traditional LLMs cannot directly reason over large PDFs.  
This project solves that by:
- Converting PDFs into semantic chunks
- Storing them in a vector database
- Retrieving relevant context at query time
- Generating accurate, grounded answers using an LLM



 Solution Overview

The system follows a standard **RAG architecture**:

1. PDF ingestion & text extraction  
2. Text chunking and embedding generation  
3. Vector storage using similarity search  
4. Context retrieval for user queries  
5. Answer generation using an LLM


Tech Stack

- **Python**
- **LangChain**
- **LLMs (OpenAI / Gemini / HuggingFace – configurable)**
- **Vector Database** (FAISS / Chroma)
- **PyPDF** for PDF parsing

#LICENSE
This project is licensced under MIT Licensesss


