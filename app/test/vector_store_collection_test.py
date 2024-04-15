from vector_store.chroma_utils import load_docs, split_docs, create_collection
import chromadb
import os

def get_absolute_path(relative_path):
    base_path = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script
    return os.path.join(base_path, relative_path)

def get_collection(collection_id:str,persistent_client):
    try:
        collection = persistent_client.get_collection(name = collection_id)
        return collection
    except ValueError as e:
        return e


persist_directory = get_absolute_path('../../app/chroma/')

persistent_client = chromadb.PersistentClient(path=persist_directory)

print(get_collection(collection_id="collection-foobar", persistent_client=persistent_client).peek())