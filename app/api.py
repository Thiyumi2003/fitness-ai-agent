from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.agent import FitnessAgent

router = APIRouter()
agent = FitnessAgent()

# In-memory user store (demo only). Replace with a real DB for production.
users_db = {}


class SignupRequest(BaseModel):
    name: str
    gender: str
    age: int
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/signup")
def signup(req: SignupRequest):
    if req.email in users_db:
        return {"success": False, "message": "Email already exists"}

    users_db[req.email] = {
        "name": req.name,
        "gender": req.gender,
        "age": req.age,
        "email": req.email,
        "password": req.password,
    }
    return {"success": True, "message": "Account created successfully"}


@router.post("/login")
def login(req: LoginRequest):
    user = users_db.get(req.email)
    if not user or user["password"] != req.password:
        return {"success": False, "message": "Invalid email or password"}

    return {
        "success": True,
        "user": {
            "name": user["name"],
            "gender": user["gender"],
            "age": user["age"],
            "email": user["email"],
        },
    }


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
    choice: str
    include_ml: Optional[bool] = False


@router.post("/profile")
def profile(profile: Profile):
    agent.state["goal"] = profile.goal
    agent.state["gender"] = profile.gender
    agent.state["weight"] = profile.weight
    agent.state["height"] = profile.height
    agent.state["age"] = profile.age
    agent.state["hypertension"] = profile.hypertension
    agent.state["diabetes"] = profile.diabetes
    agent.state["fitness_type"] = profile.fitness_type
    agent.state["choice"] = profile.choice

    try:
        bmi_msg = agent.calculate_bmi()
    except Exception:
        bmi_msg = "BMI unavailable"

    try:
        plan = agent.get_plan_from_dataset(profile.choice)
    except Exception:
        plan = "No dataset plan available."

    ml_rec = None
    if profile.include_ml:
        try:
            if profile.choice.lower().startswith("ex"):
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
