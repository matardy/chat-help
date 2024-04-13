from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 

load_dotenv()
openai_model = os.getenv('OPENAI_MODEL')

chat_model = ChatOpenAI(model=openai_model, temperature=0)