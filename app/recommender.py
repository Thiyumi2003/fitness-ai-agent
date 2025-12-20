import pandas as pd

df = pd.read_csv("data/gym_data.csv")

def get_recommendation(goal, bmi, sex):

    goal = goal.replace("_", " ").lower()
    sex = sex.lower()

    results = df[
        (df["Fitness Goal"].str.lower().str.contains(goal)) &
        (df["Sex"].str.lower().str.contains(sex))
    ]

    if results.empty:
        return "⚠️ No suitable plan found for your profile."

    row = results.iloc[0]

    response = f"""
🏋️ Exercises:
{row['Exercises']}

🧰 Equipment:
{row['Equipment']}

🥗 Diet:
{row['Diet']}

📌 Recommendation:
{row['Recommendation']}
"""
    return response.strip()
