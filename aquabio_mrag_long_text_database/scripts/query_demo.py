import sys
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "data" / "vector_db" / "chroma_text"
COLLECTION = "aquabio_text"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "海星有什么视觉特征？"
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_collection(COLLECTION)

    emb = model.encode([query], normalize_embeddings=True).tolist()[0]
    res = col.query(query_embeddings=[emb], n_results=5)

    print("QUERY:", query)
    for i, doc in enumerate(res["documents"][0], 1):
        md = res["metadatas"][0][i-1]
        print("\n---", i, "---")
        print("species_id:", md.get("species_id"))
        print("chunk_type:", md.get("chunk_type"))
        print(doc[:500])

if __name__ == "__main__":
    main()
