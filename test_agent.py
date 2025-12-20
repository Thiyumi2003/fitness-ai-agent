from app.agent import FitnessAgent

agent = FitnessAgent()

while True:
    user_input = input("You: ")
    response = agent.process(user_input)
    print("Agent:", response)
