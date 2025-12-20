from app.agent import FitnessAgent

agent = FitnessAgent()
inputs = [
    "i want weight loss",
    "Female",
    "50",
    "1.6",
    "22",
    "No",
    "No",
    "Diet plan",
    "Cardio Fitness",
    "Yes"
]
for msg in inputs:
    print("You:", msg)
    resp = agent.process(msg)
    print("Agent:", resp)
