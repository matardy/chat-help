from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from chains.llm import chat_model
from cache.message_history import get_memory_runnable
from vector_store.chroma_utils import get_retriever
from langchain.schema.runnable import RunnablePassthrough
from vector_store.chroma_utils import load_docs, split_docs, create_collection
import chromadb
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Any
def get_absolute_path(relative_path):
    base_path = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script
    return os.path.join(base_path, relative_path)

persist_directory = get_absolute_path('/app/chroma/')
persistent_client = chromadb.PersistentClient(path=persist_directory)
vector_db = Chroma(
        client=persistent_client,
        persist_directory=persist_directory,
        collection_name="collection-foobar",
        embedding_function=OpenAIEmbeddings(),
        collection_metadata={"hnsw:space": "cosine"}
    )
retriever = vector_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.56},
    k=3
)

review_template_str = """You are an Legal Assistant AI chat, you have full context of the conversation and always respond in a very elegant way. 
- Always respond in the language of the user. 
- If you dont know the answer, just say "Plase can you be more specific?" 
- Your main job is to assist lawyer to get answers.
- You have to follow the conversation and answer follow up questions like: Be more specific, what? , continue, etc. 

You should respond base on the context provide here:
{context}

Message histry:
{chat_history}
"""

system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context","chat_history"],
        template=review_template_str
    )
)

human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["input"],
        template = "{input}"
    )
)

messages = [system_prompt, human_prompt]

prompt_template = ChatPromptTemplate(
    input_variables=["context", "input","chat_history"], 
    messages=messages,
)

# Contextualize question 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    chat_model, retriever, contextualize_q_prompt
)


qa_chain = prompt_template | chat_model | StrOutputParser()
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# chain = (
#     {"context":retriever , "question": RunnablePassthrough()}
#     | prompt_template 
#     | chat_model
#     | StrOutputParser()
# )  
chain = get_memory_runnable(rag_chain)

# TODO: New chat sessions needs to have diferents session_id
def model_response(question:str) -> Dict[str, Any]:
    response = chain.invoke({"input": question},
    config={
        "configurable":{"session_id": "foobar"}
    })
    docs_array = response.get('context')
    documents_json = [
        {
            "page_content": doc.page_content,
            "page": doc.metadata['page'],
            "source": doc.metadata['source']
        } for doc in docs_array
    ]

    print("Context on this query: ", documents_json)
    return {"answer": response.get('answer'), "context": documents_json}

