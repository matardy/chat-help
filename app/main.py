from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv


from chains.chain import model_response

# Initialize FastAPI app
app = FastAPI()
load_dotenv()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class ChatRequest(BaseModel):
    message: str

@app.post("/message/")
async def chat(chat_request: ChatRequest):
    response = model_response(question=chat_request.message)
    return {"response": response}