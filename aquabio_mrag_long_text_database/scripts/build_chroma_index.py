import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DOC_FILE = ROOT / "data" / "knowledge" / "rag_documents_combined_text_only.jsonl"
DB_DIR = ROOT / "data" / "vector_db" / "chroma_text"
COLLECTION = "aquabio_text"

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
# 如果下载较慢，可改成：
# MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def main():
    docs = read_jsonl(DOC_FILE)
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION)

    ids = [d["id"] for d in docs]
    texts = [d["embedding_text"] for d in docs]
    contents = [d["content"] for d in docs]
    metadatas = []
    for d in docs:
        md = d.get("metadata", {})
        md["species_id"] = d.get("species_id", "")
        md["source_type"] = d.get("source_type", "")
        md["modality"] = d.get("modality", "text")
        metadatas.append(md)

    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()

    batch = 64
    for i in range(0, len(ids), batch):
        col.add(
            ids=ids[i:i+batch],
            documents=contents[i:i+batch],
            metadatas=metadatas[i:i+batch],
            embeddings=embeddings[i:i+batch]
        )

    print(f"Built Chroma collection: {COLLECTION}")
    print(f"Docs: {len(ids)}")
    print(f"DB: {DB_DIR}")

if __name__ == "__main__":
    main()
