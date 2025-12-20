from fastapi import APIRouter
from pydantic import BaseModel
from app.agent import FitnessAgent

router = APIRouter()
agent = FitnessAgent()

class UserInput(BaseModel):
    message: str

@router.post("/chat")
def chat(user_input: UserInput):
    response = agent.process(user_input.message)
    return {"reply": response}
