import os
import json
import fitz  
import openai
import numpy as np
import faiss

CHUNK_SIZE = 50
EMBED_MODEL = "text-embedding-3-small"
INDEX_FILE_PATH = "faiss.index"
CHUNKS_FILE_PATH = "chunks_store.json"
openai.api_key = "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"
index = None
chunks = []


def load_chunks():
    global chunks
    if os.path.exists(CHUNKS_FILE_PATH):
        with open(CHUNKS_FILE_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        chunks = []


def save_chunks():
    with open(CHUNKS_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f)


def embed_chunks(chunk_list):
    embeddings = openai.embeddings.create(model=EMBED_MODEL, input=chunk_list)
    vectors = np.array([e.embedding for e in embeddings.data]).astype("float32")
    return vectors


def chunk_text(text):
    words = text.split()
    return [" ".join(words[i: i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]


def extract_text_from_pdf(pdf_path):
    pages = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in pages])
    return full_text


def add_document():
    global index, chunks

    pdf_path = input("Enter path of PDF file: ").strip()
    if not os.path.exists(pdf_path):
        print(f"File not found at {pdf_path}")
        return

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("No content found. Try another PDF file.")
        return

    new_chunks = chunk_text(text)
    new_vectors = embed_chunks(new_chunks)

    if os.path.exists(INDEX_FILE_PATH):
        index = faiss.read_index(INDEX_FILE_PATH)
    else:
        index = faiss.IndexFlatL2(new_vectors.shape[1])

    index.add(new_vectors)
    faiss.write_index(index, INDEX_FILE_PATH)

    chunks.extend(new_chunks)
    save_chunks()

    print("Document indexed successfully.")


def search_faiss_index(query_vector, index):
    distance, indices = index.search(query_vector, 5)
    return indices[0]


def ask_gpt4(context, query):
    context = "\n".join(context)

    system_prompt = (
        "You are an expert assistant. Answer only based on the provided context. "
        "If the answer is not found, say 'No information available'."
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
        print(" Document index not found. Use option 1 to add documents.")
        return

    index = faiss.read_index(INDEX_FILE_PATH)
    load_chunks()

    query = input("Enter your question: ").strip()
    if not query:
        print("Query cannot be empty.")
        return

    query_embedding = openai.embeddings.create(model=EMBED_MODEL, input=query)
    query_vector = np.array([e.embedding for e in query_embedding.data]).astype("float32")

    top_indices = search_faiss_index(query_vector, index)
    context = [chunks[i] for i in top_indices if i < len(chunks)]

    if not context:
        print(" No matching context found.")
        return

    answer = ask_gpt4(context, query)
    print(f"\n Answer for: {query}\n➡ {answer}\n")


def delete_document():
    confirm = input("Are you sure you want to delete all indexed data? (yes/no): ").strip().lower()
    if confirm == "yes":
        if os.path.exists(INDEX_FILE_PATH):
            os.remove(INDEX_FILE_PATH)
        if os.path.exists(CHUNKS_FILE_PATH):
            os.remove(CHUNKS_FILE_PATH)
        print("🗑️ Index and data deleted.")
    else:
        print(" Deletion canceled.")


def main():
    while True:
        print("\n📘 Select an Option:")
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
            print("👋 Goodbye.")
            break
        else:
            print(" Invalid choice. Try again.")


if __name__ == "__main__":
    main()
