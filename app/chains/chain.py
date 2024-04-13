from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from chains.llm import chat_model
from cache.message_history import get_memory_runnable

review_template_str = """You are an Legal Assistant AI chat, you have full context of the conversation and always respond in a very elegant way. 
- Always respond in the language of the user. 
- If you dont know the answer, just say "Plase can you be more specific?" 
- Your main job is to assist lawyer to get answers.
- You have to follow the conversation and answer follow up questions like: Be more specific, what? , continue, etc. 
{context}

Message histry:
{history}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context","history"],
        template=review_template_str
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template = "{question}"
    )
)

messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question","history"], 
    messages=messages,
)

output_parser = StrOutputParser()

review_chain = review_prompt_template | chat_model | output_parser
review_chain = get_memory_runnable(review_chain)

def model_response(context: str, question:str) -> str:
    return review_chain.invoke({
        "context": context,
        "question": question,
    },
    # TODO: New chat sessions needs to have diferents session_id
    config={
        "configurable":{"session_id": "foobar"}
    })
    