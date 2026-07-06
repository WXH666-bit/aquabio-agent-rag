import shutil
from pathlib import Path

vector_dir = Path(r"C:\Users\18246\Desktop\homework\data\mrag\vector_db\chroma")
if vector_dir.exists():
    shutil.rmtree(vector_dir)
    print("ChromaDB directory deleted")
else:
    print("Directory not found")
vector_dir.mkdir(parents=True, exist_ok=True)
print("Empty directory created, ready for rebuild")