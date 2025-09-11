import os
import numpy as np
import faiss
import pickle
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- Initialization ---
load_dotenv()

# Configure Flask App
app = Flask(__name__)
# Allow cross-origin requests from your specific portfolio URL for security
CORS(app, resources={r"/ask": {"origins": "https://ashvinmanojk289.github.io"}})

# Configure Gemini API
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    gemini_model = genai.GenerativeModel(
        model_name="gemini-1.0-pro",
        generation_config=generation_config
    )
    print("Gemini model loaded successfully.")
except Exception as e:
    print(f"Error loading Gemini model: {e}")
    gemini_model = None

# Load Sentence Transformer Model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence Transformer model loaded successfully.")
except Exception as e:
    print(f"Error loading Sentence Transformer model: {e}")
    embedding_model = None

# Load FAISS index and text chunks
try:
    faiss_index = faiss.read_index('resume_index.faiss')
    with open('resume_chunks.pkl', 'rb') as f:
        text_chunks = pickle.load(f)
    print("FAISS index and text chunks loaded successfully.")
except Exception as e:
    print(f"Error loading knowledge base: {e}")
    faiss_index = None
    text_chunks = None

# --- API Endpoint ---
@app.route('/ask', methods=['POST'])
def ask_assistant():
    if not all([gemini_model, embedding_model, faiss_index, text_chunks]):
        return jsonify({"error": "Assistant is not properly configured. Check server logs."}), 500

    data = request.get_json()
    user_question = data.get('question')

    if not user_question:
        return jsonify({"error": "No question provided."}), 400

    # 1. Embed the user's question
    question_embedding = embedding_model.encode([user_question]).astype('float32')

    # 2. Retrieve relevant chunks from FAISS
    k = 3 # Number of relevant chunks to retrieve
    distances, indices = faiss_index.search(question_embedding, k)
    
    retrieved_chunks = [text_chunks[i] for i in indices[0]]
    context = "\n---\n".join(retrieved_chunks)

    # 3. Augment and Generate with Gemini
    prompt = f"""You are a professional and friendly AI assistant for Ashvin Manoj's portfolio.
    Your task is to answer questions about him based ONLY on the provided context from his resume.
    If the answer is not in the context, politely state that you don't have that information from his resume.
    Do not make up information. Be concise, helpful, and speak in the first person as if you are Ashvin's assistant.

    CONTEXT FROM RESUME:
    {context}

    USER'S QUESTION:
    {user_question}

    ANSWER:"""

    try:
        response = gemini_model.generate_content(prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        print(f"Error during Gemini generation: {e}")
        return jsonify({"error": "Failed to generate a response from the AI model."}), 500

# A simple health check endpoint
@app.route('/')
def health_check():
    return "AI Assistant backend is running."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)