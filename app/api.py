from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.agent import FitnessAgent

router = APIRouter()
agent = FitnessAgent()

class UserInput(BaseModel):
    message: str

@router.post("/chat")
def chat(user_input: UserInput):
    response = agent.process(user_input.message)
    return {"reply": response}


class Profile(BaseModel):
    goal: str
    gender: str
    weight: float
    height: float
    age: int
    hypertension: str
    diabetes: str
    fitness_type: Optional[str] = None
    choice: str  # 'exercise' or 'diet'
    include_ml: Optional[bool] = False


@router.post("/profile")
def profile(profile: Profile):
    # populate agent state from provided profile
    agent.state["goal"] = profile.goal
    agent.state["gender"] = profile.gender
    agent.state["weight"] = profile.weight
    agent.state["height"] = profile.height
    agent.state["age"] = profile.age
    agent.state["hypertension"] = profile.hypertension
    agent.state["diabetes"] = profile.diabetes
    agent.state["fitness_type"] = profile.fitness_type
    agent.state["choice"] = profile.choice

    # compute BMI and category
    try:
        bmi_msg = agent.calculate_bmi()
    except Exception:
        bmi_msg = "BMI unavailable"

    # dataset plan
    try:
        plan = agent.get_plan_from_dataset(profile.choice)
    except Exception:
        plan = "No dataset plan available."

    ml_rec = None
    if profile.include_ml:
        try:
            if profile.choice.lower().startswith('ex'):
                ml_rec = agent.ml_exercise_recommendation()
            else:
                ml_rec = agent.ml_diet_recommendation()
        except Exception:
            ml_rec = "ML recommendation unavailable (models may be missing)."

    return {
        "bmi_message": bmi_msg,
        "plan": plan,
        "ml_recommendation": ml_rec,
    }
