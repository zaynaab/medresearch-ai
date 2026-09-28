# MedResearch AI

An AI-powered medical research assistant that uses Large Language Models (LLMs), embeddings, semantic retrieval, and Retrieval-Augmented Generation (RAG) to answer questions from research documents.

## Overview

MedResearch AI is a learning-focused implementation of a Retrieval-Augmented Generation pipeline.

The current version processes a medical research paper, divides the document into chunks, converts the chunks into embeddings, retrieves the most semantically relevant sections for a user question, and provides the retrieved context to a Gemini language model to generate an answer.

## Current Architecture

```text
Research PDF
     ↓
Text Extraction
     ↓
Document Chunking
     ↓
Embedding Generation
     ↓
Embedding Storage
     ↓
Question Embedding
     ↓
Cosine Similarity Search
     ↓
Top-K Relevant Chunks
     ↓
Gemini LLM
     ↓
Generated Answer


Technologies
Python
Google Gemini API
Gemini Embeddings
PyMuPDF
NumPy / Python similarity calculations
Retrieval-Augmented Generation (RAG)
Semantic Search
Current Features
PDF text extraction
Paragraph-aware document chunking
Text embeddings using Gemini
Persistent embedding storage
Semantic similarity search
Top-k document retrieval
LLM-based answer generation using retrieved context

PROJECT STRUCTURE
medresearch-ai/
│
├── documents/
│   └── README.md
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

***SETUP***
1) CLONE THE REPOSITORY
git clone <your-repository-url>
cd medresearch-ai

2) Create a virtual environment
python -m venv .venv

3)Install dependencies
pip install -r requirements.txt

4) Configure the API key
GEMINI_API_KEY=your_api_key_here

5)Add a research document
documents/

6) RUN
python main.py

Learning Objectives

This project was built to understand the core mechanics behind modern RAG systems without relying on high-level frameworks.

The implementation currently demonstrates:

Document ingestion
Text chunking
Embedding generation
Vector similarity
Semantic retrieval
Context augmentation
LLM generation
Roadmap

The system will be progressively developed into a more production-oriented AI application.

 FAISS vector search
 Improved document chunking
 Metadata management
 Source citations
 Multi-document retrieval
 Retrieval evaluation
 Reranking
 FastAPI backend
 Dockerization
 Deployment
 Monitoring and logging
Disclaimer

This project is an AI/ML engineering and research demonstration. It is not intended to provide medical diagnosis or clinical advice.
