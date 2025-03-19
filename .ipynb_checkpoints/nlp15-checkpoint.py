from fastapi import FastAPI
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOllama

app = FastAPI()

# Add a simple home route
@app.get("/")
def home():
    return {"message": "Welcome to my FastAPI chatbot!"}

# Initialize chatbot
chatbot = ChatOllama(model='mistral')
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=chatbot,
    memory=memory
)

class ChatRequest(BaseModel):
    user_input: str

@app.post('/chat')
def chat(request: ChatRequest):
    user_input = request.user_input
    response = conversation.predict(input=user_input)
    return {'response': response}
