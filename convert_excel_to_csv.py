import pandas as pd

# Read Excel file
df = pd.read_excel("data/gym recommendation.xlsx")

# Convert to CSV
df.to_csv("data/gym_data.csv", index=False)

print("✅ Excel converted to CSV successfully!")
