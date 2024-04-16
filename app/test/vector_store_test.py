from vector_store.chroma_utils import load_docs, split_docs, create_collection
import chromadb
import os

def get_absolute_path(relative_path):
    base_path = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script
    return os.path.join(base_path, relative_path)

persist_directory = get_absolute_path('../../app/chroma/')
docs_path = [get_absolute_path('../../app/docs/niif-16-arrendamientos-1.pdf'),
             get_absolute_path('../../app/docs/niif-16-arrendamientos-2.pdf'),
             get_absolute_path('../../app/docs/niif-16-arrendamientos.pdf')]

persistent_client = chromadb.PersistentClient(path=persist_directory)
for doc in docs_path:
    doc = load_docs(doc)

    print("Loaded docs: ", doc, "\n")

    split = split_docs(chunk_size=1500, chunk_overlap=150, docs=doc)

    print("Loaded splits: ", split, "\n")

    collection, chunks = create_collection(
                        collection_id="collection-foobar", 
                        splits=split,
                        persistent_client=persistent_client
                        )

    print("Collection: ", collection)

