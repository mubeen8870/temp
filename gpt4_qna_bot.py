import os
import fitz
import openai
import numpy as np
import faiss
import pickle
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CHUNK_SIZE = 50
EMBED_MODEL = "text-embedding-3-small"
INDEX_FILE_PATH = "faiss.index"
CHUNKS_FILE = "chunks.pkl"

index = None
chunks = []

def embed_chunks(chunks):
    embeddings = openai.embeddings.create(model=EMBED_MODEL, input=chunks)
    vectors = np.array([e.embedding for e in embeddings.data]).astype("float32")
    return vectors

def chunk_text(text):
    words = text.split()
    return [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

def extract_text_from_pdf(pdf_path):
    pages = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in pages])
    return full_text

def save_chunks(chunks):
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

def load_chunks():
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "rb") as f:
            return pickle.load(f)
    return []

def add_document():
    global index, chunks
    pdf_path = input("Enter path of PDF file : ").strip()
    if not os.path.exists(pdf_path):
        print(f"File not found at {pdf_path}")
        return

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("No content found. Try another PDF file.")
        return

    new_chunks = chunk_text(text)
    vectors = embed_chunks(new_chunks)

    if not os.path.exists(INDEX_FILE_PATH):
        index = faiss.IndexFlatL2(vectors.shape[1])
    else:
        index = faiss.read_index(INDEX_FILE_PATH)

    index.add(vectors)
    faiss.write_index(index, INDEX_FILE_PATH)

    chunks = load_chunks()
    chunks.extend(new_chunks)
    save_chunks(chunks)
    print("Document was indexed successfully.")

def search_faiss_index(query_vector, index):
    distance, indices = index.search(query_vector, 5)
    return indices[0]

def ask_gpt4(context, query):
    context = "\n".join(context)
    system_prompt = (
        "You are an expert assistant. Answer only based on the provided context. "
        "If the answer is not found then say 'No information available'."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\nQuestion: {query}"}
    ]

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def query_document():
    global index, chunks

    if not os.path.exists(INDEX_FILE_PATH):
        print("Document index does not exist. Please choose option 1 to create it.")
        return

    index = faiss.read_index(INDEX_FILE_PATH)
    chunks = load_chunks()

    query = input("Enter your query to search the index: ").strip()
    if not query:
        print("Query cannot be empty.")
        return

    query_embedding = openai.embeddings.create(model=EMBED_MODEL, input=query)
    query_vector = np.array([e.embedding for e in query_embedding.data]).astype("float32")

    top_indices = search_faiss_index(query_vector, index)
    context = [chunks[i] for i in top_indices if i < len(chunks)]

    if not context:
        print("No relevant information found.")
        return

    answer = ask_gpt4(context, query)
    print(f"\nAnswer: {answer}")

def delete_document():
    confirm = input("Are you sure you want to delete all indexed data? (yes/no): ").strip().lower()
    if confirm == "yes":
        if os.path.exists(INDEX_FILE_PATH):
            os.remove(INDEX_FILE_PATH)
        if os.path.exists(CHUNKS_FILE):
            os.remove(CHUNKS_FILE)
        print("Index and chunks have been deleted.")
    else:
        print("Deletion canceled.")

def main():
    while True:
        print("\nSelect an Option:")
        print("1. Add Document to FAISS Index")
        print("2. Query Document")
        print("3. Delete Document")
        print("4. Exit")
        choice = input("Please select an option (1/2/3/4): ").strip()

        if choice == "1":
            add_document()
        elif choice == "2":
            query_document()
        elif choice == "3":
            delete_document()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
