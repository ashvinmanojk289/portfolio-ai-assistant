import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# --- 1. Document Loading and Chunking ---
print("Reading resume PDF...")
# This script assumes 'resume.pdf' is in the same folder.
pdf_path = 'resume.pdf'
resume_text = ""

try:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            resume_text += page.extract_text()
except FileNotFoundError:
    print(f"Error: The file '{pdf_path}' was not found. Please copy your resume to this folder.")
    exit()

# A simple chunking strategy to split the text into smaller parts.
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

text_chunks = chunk_text(resume_text)
print(f"Resume successfully split into {len(text_chunks)} chunks.")

# --- 2. Embedding Generation ---
print("Loading sentence transformer model (this may take a moment)...")
# Using a powerful but small model perfect for this task
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating embeddings for text chunks...")
embeddings = model.encode(text_chunks, show_progress_bar=True)
# Ensure embeddings are float32, as required by FAISS
embeddings = np.array(embeddings).astype('float32')

# --- 3. FAISS Index Creation ---
print("Creating FAISS index for similarity search...")
embedding_dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)
index.add(embeddings)
print(f"FAISS index created with {index.ntotal} vectors.")

# --- 4. Saving the Index and Chunks for the Server to Use ---
# We need to save both the index (for searching) and the original text chunks (for context).
faiss.write_index(index, 'resume_index.faiss')

with open('resume_chunks.pkl', 'wb') as f:
    pickle.dump(text_chunks, f)

print("\nProcessing complete!")
print("Knowledge base created and saved as 'resume_index.faiss' and 'resume_chunks.pkl'.")
print("You can now run the 'app.py' server.")