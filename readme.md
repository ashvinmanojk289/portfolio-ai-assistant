# Portfolio AI Assistant Backend

This repository contains the backend code for the AI assistant featured on my personal portfolio website.

## Overview

This is a Python-based web server built with Flask that uses a Retrieval-Augmented Generation (RAG) architecture to answer questions about my resume.

-   **Knowledge Base**: The text content from my resume (`resume.pdf`).
-   **Embedding Model**: `all-MiniLM-L6-v2` from Hugging Face's `sentence-transformers`.
-   **Vector Store**: `FAISS` for efficient similarity search.
-   **LLM**: Google's `Gemini-1.0-Pro` for generating natural language responses.

## Setup and Running

### 1. Initial Setup (Run Once)

To create the knowledge base, you must run the processing script first.

1.  Place your resume in this folder and name it `resume.pdf`.
2.  Install all required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the processing script:
    ```bash
    python process_resume.py
    ```
    This will generate `resume_index.faiss` and `resume_chunks.pkl`.

### 2. Running the Server

1.  Create a `.env` file and add your `GOOGLE_API_KEY`.
2.  Start the Flask server for local testing:
    ```bash
    python app.py
    ```
3.  For production, use gunicorn:
    ```bash
    gunicorn app:app
    ```
The server will be running on `http://127.0.0.1:5000`.