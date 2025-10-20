from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from data_format import chunk_text, extract_pdf


model = SentenceTransformer("all-MiniLM-L6-v2")


def open_db(db_path: str = "db"):
    """Open (or create) the persistent Chroma collection for PDF chunks."""
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name="pdf_chunks")


def make_db(chunks, db_path: str = "db"):
    """Create and store embeddings in a ChromaDB collection without duplicates."""
    collection = open_db(db_path)

    # Determine which ids already exist to avoid duplicate adds
    existing_items = collection.get(include=[])
    existing_ids = set(existing_items.get("ids", []))

    new_indices = [i for i in range(len(chunks)) if f"chunk_{i}" not in existing_ids]
    if not new_indices:
        print(f"No new chunks to add. Using existing ChromaDB at '{db_path}'")
        return collection

    new_chunks = [chunks[i] for i in new_indices]
    new_embeddings = model.encode(new_chunks).tolist()
    new_ids = [f"chunk_{i}" for i in new_indices]

    collection.add(documents=new_chunks, embeddings=new_embeddings, ids=new_ids)
    print(f"Stored {len(new_chunks)} new chunks in ChromaDB at '{db_path}'")
    return collection


def query_collection(collection, question: str, k: int = 4):
    """Query the collection using semantic similarity and return top-k documents."""
    query_embedding = model.encode([question]).tolist()
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "distances", "metadatas"],  # 'ids' is not a valid include in Chroma
    )
    documents = result.get("documents", [[]])[0]
    distances = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])
    ids = ids[0] if ids else [None] * len(documents)
    if not ids or len(ids) != len(documents):
        ids = [None] * len(documents)
    return [{"id": i, "text": d, "score": dist} for i, d, dist in zip(ids, documents, distances)]

