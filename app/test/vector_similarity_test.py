
import chromadb
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def get_absolute_path(relative_path):
    base_path = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script
    return os.path.join(base_path, relative_path)

persist_directory = get_absolute_path('../../app/chroma/')
persistent_client = chromadb.PersistentClient(path=persist_directory)
vector_db = Chroma(
        client=persistent_client,
        persist_directory=persist_directory,
        collection_name="collection-foobar",
        embedding_function=OpenAIEmbeddings()
    )
question = " Cual es el tratamiento contable si actualmente mantengo un contrato de arrendamiento de un panel solar, en el que pago una cuota fija de 10000 USD y por otro lado una cuota variable del 10% del ahorro del costo de kilovatio hora que mantenía antes de la implementar los paneles solares vs el pago mensual de arrendamiento. El periodo de tiempo es 20 años de arrendamiento."
# retriever = vector_db.as_retriever(k=10)
relevant_docs = vector_db.similarity_search(question, k=3)
print(relevant_docs[0].page_content)