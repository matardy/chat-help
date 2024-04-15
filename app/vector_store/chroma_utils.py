from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import uuid
import os
import openai
from chromadb.utils import embedding_functions

from langchain_openai import OpenAIEmbeddings
import chromadb
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
# Persist directory : '/app/chroma/'


openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small",
    api_key=openai.api_key
)

def load_docs(docs_paths):
    docs = []
    loader = PyPDFLoader(docs_paths)
    docs.extend(loader.load())
    return docs

def split_docs(chunk_size:int, chunk_overlap:int, docs:list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    return text_splitter.split_documents(docs)

def create_collection(collection_id:str, splits:list, persistent_client,embedding_function = openai_ef):
    # Guarda los chunks id de cada split
    chunks_id = []
    collection = persistent_client.get_or_create_collection(name = collection_id,embedding_function = embedding_function)

    for split in splits:
        id_ = [str(uuid.uuid1())]
        chunks_id.append(id_[0])
        collection.add(
            ids=id_, metadatas=split.metadata, documents=split.page_content
        )
    return collection, chunks_id

def get_absolute_path(relative_path):
    base_path = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script
    return os.path.join(base_path, relative_path)

persist_directory = get_absolute_path('../../app/chroma/')
persistent_client = chromadb.PersistentClient(path=persist_directory)
def get_retriever():
    vector_db = Chroma(
        client=persistent_client,
        persist_directory=persist_directory,
        collection_name="collection-foobar",
        embedding_function=OpenAIEmbeddings()
    )
    return vector_db.as_retriever(k=10)
