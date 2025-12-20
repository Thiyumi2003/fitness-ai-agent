from dataclasses import field
import pandas as pd
import joblib

class FitnessAgent:
    def __init__(self):
        # Conversation state
        self.state = {
            "goal": None,
            "gender": None,
            "weight": None,
            "height": None,
            "age": None,
            "hypertension": None,
            "diabetes": None,
            "fitness_type": None,
            "choice": None,
            "awaiting_recommendation": None,
            "bmi": None,
            "bmi_category": None
        }

        # Load ML models and separate encoders (debug prints added)
       
        self.ml_ex_model = joblib.load("app/exercise_model.pkl")
        
        self.ml_ex_encoders = joblib.load("app/exercise_encoders.pkl")
        
        self.ml_diet_model = joblib.load("app/diet_model.pkl")
        
        self.ml_diet_encoders = joblib.load("app/diet_encoders.pkl")
        
        # Load dataset for fallback
        
        self.df = pd.read_csv("data/gym_data.csv")
        
    # Normalize user inputs to match ML training labels
    def normalize_inputs(self, value, field):
        # Defensive: handle None
        if value is None:
            return ""

        if field == "Sex":
            v = value.strip().lower()
            if v in ["female", "f"]:
                return "Female"
            if v in ["male", "m"]:
                return "Male"
            return value.strip().title()

        if field in ["Hypertension", "Diabetes"]:
            v = value.strip().lower()
            if v in ["yes", "y"]:
                return "Yes"
            if v in ["no", "n"]:
                return "No"
            return value.strip().title()

        if field == "Fitness Goal":
            v = value.strip()
            title_v = v.title()
            try:
                if title_v in self.df["Fitness Goal"].unique():
                    return title_v
            except Exception:
                pass
            low = v.lower()
            if "weight" in low and ("loss" in low or "lose" in low):
                return "Weight Loss"
            if "weight" in low and "gain" in low:
                return "Weight Gain"
            if "lose" in low or "loss" in low:
                return "Weight Loss"
            if "gain" in low:
                return "Weight Gain"
            return title_v

        if field == "Level":
            # normalize bmi categories to dataset Level values
            v = value.strip().title()
            # dataset has a typo 'Obuse' — map 'Obese' to 'Obuse' if needed
            if v == "Obese":
                # prefer dataset value if present
                try:
                    if "Obuse" in self.df["Level"].unique():
                        return "Obuse"
                except Exception:
                    pass
            return v

        if field == "Fitness Type":
            v = value.strip().title()
            try:
                if v in self.df["Fitness Type"].unique():
                    return v
            except Exception:
                pass
            # map common keywords
            low = value.strip().lower()
            if "cardio" in low:
                return "Cardio Fitness"
            if "muscle" in low or "muscular" in low or "strength" in low:
                return "Muscular Fitness"
            return v

        return value

    # BMI calculation
    def calculate_bmi(self):
        w = self.state["weight"]
        h = self.state["height"]
        bmi = w / (h ** 2)
        self.state["bmi"] = round(bmi, 2)
        if bmi < 18.5:
            self.state["bmi_category"] = "Underweight"
        elif 18.5 <= bmi < 25:
            self.state["bmi_category"] = "Normal"
        elif 25 <= bmi < 30:
            self.state["bmi_category"] = "Overweight"
        else:
            self.state["bmi_category"] = "Obese"
        return f"Your BMI is {self.state['bmi']} ({self.state['bmi_category']})."

    # ML Exercise Recommendation
    def ml_exercise_recommendation(self):
        data = {
            "Sex": self.normalize_inputs(self.state["gender"], "Sex"),
            "Age": self.state["age"],
            "BMI": self.state["bmi"],
            "Fitness Goal": self.normalize_inputs(self.state["goal"], "Fitness Goal"),
            "Level": self.normalize_inputs(self.state["bmi_category"], "Level"),
            "Hypertension": self.normalize_inputs(self.state["hypertension"], "Hypertension"),
            "Diabetes": self.normalize_inputs(self.state["diabetes"], "Diabetes")
        }
        # include Fitness Type only if encoder/model expects it
        if "Fitness Type" in self.ml_ex_encoders:
            data["Fitness Type"] = self.normalize_inputs(self.state.get("fitness_type", ""), "Fitness Type")
        df_input = pd.DataFrame([data])
        # Safe transform: ensure value exists in encoder classes, attempt case-insensitive match, fallback to first class
        for col, le in self.ml_ex_encoders.items():
            val = str(df_input[col].iloc[0])
            if val in le.classes_:
                df_input[col] = le.transform(df_input[col].astype(str))
                continue
            # try case-insensitive match
            matched = None
            for cls in le.classes_:
                if cls.lower() == val.lower():
                    matched = cls
                    break
            if matched is not None:
                df_input[col] = le.transform([matched])
                continue
            # fallback: if encoder has numeric mapping for unknowns, choose first class
            df_input[col] = le.transform([le.classes_[0]])
        prediction = self.ml_ex_model.predict(df_input)[0]
        return f"🏋️ ML Suggested Exercise Plan:\n{prediction}"

    # ML Diet Recommendation
    def ml_diet_recommendation(self):
        data = {
            "Sex": self.normalize_inputs(self.state["gender"], "Sex"),
            "Age": self.state["age"],
            "BMI": self.state["bmi"],
            "Fitness Goal": self.normalize_inputs(self.state["goal"], "Fitness Goal"),
            "Level": self.normalize_inputs(self.state["bmi_category"], "Level"),
            "Hypertension": self.normalize_inputs(self.state["hypertension"], "Hypertension"),
            "Diabetes": self.normalize_inputs(self.state["diabetes"], "Diabetes")
        }
        # include Fitness Type only if encoder/model expects it
        if "Fitness Type" in self.ml_diet_encoders:
            data["Fitness Type"] = self.normalize_inputs(self.state.get("fitness_type", ""), "Fitness Type")
        df_input = pd.DataFrame([data])
        for col, le in self.ml_diet_encoders.items():  # use Diet encoder
            val = str(df_input[col].iloc[0])
            if val in le.classes_:
                df_input[col] = le.transform(df_input[col].astype(str))
                continue
            matched = None
            for cls in le.classes_:
                if cls.lower() == val.lower():
                    matched = cls
                    break
            if matched is not None:
                df_input[col] = le.transform([matched])
                continue
            df_input[col] = le.transform([le.classes_[0]])
        prediction = self.ml_diet_model.predict(df_input)[0]
        return f"🥗 ML Suggested Diet Plan:\n{prediction}"

    def get_plan_from_dataset(self, choice: str):
        # try to find a matching row in dataset and return exercises/equipment or diet/recommendation
        goal = self.normalize_inputs(self.state.get("goal", ""), "Fitness Goal")
        ftype = self.normalize_inputs(self.state.get("fitness_type", ""), "Fitness Type")
        level = self.normalize_inputs(self.state.get("bmi_category", ""), "Level")
        df = self.df
        # prefer exact matches
        cond = (df["Fitness Goal"] == goal)
        if ftype:
            cond = cond & (df["Fitness Type"] == ftype)
        if level:
            cond = cond & (df["Level"] == level)
        matches = df[cond]
        # fallback to goal-only
        if matches.empty:
            matches = df[df["Fitness Goal"] == goal]
        # fallback to fitness type
        if matches.empty and ftype:
            matches = df[df["Fitness Type"] == ftype]

        if matches.empty:
            return "No dataset plan found. Would you like a personalized ML recommendation instead?"

        row = matches.iloc[0]

        def split_items(s: str):
            if not s or not isinstance(s, str):
                return []
            # normalize separators
            s2 = s.replace(" and ", ", ")
            parts = [p.strip() for p in s2.replace(";", ",").split(",")]
            return [p for p in parts if p]

        def format_lines(items):
            if not items:
                return ""
            return "\n" + "\n".join(items)

        if choice == "exercise":
            exercises = row.get("Exercises", "")
            equipment = row.get("Equipment", "")
            rec = row.get("Recommendation", "")

            ex_items = split_items(exercises)
            eq_items = split_items(equipment)

            text = "Exercise Plan:"
            if ex_items:
                text += "\nExercises:" + format_lines(ex_items)
            else:
                text += "\nExercises: No exercises listed"

            if eq_items:
                text += "\n\nEquipment:" + format_lines(eq_items)
            else:
                text += "\n\nEquipment: No equipment listed"

            if isinstance(rec, str) and rec:
                text += f"\n\nNotes:\n{rec}"
            return text
        else:
            diet = row.get("Diet", "")
            rec = row.get("Recommendation", "")

            text = "Diet Plan:"

            # Preferred formatting: show Vegetables, then Protein Intake, then Juice each on its own block
            if isinstance(diet, str) and diet:
                labels = ["Vegetables:", "Protein Intake:", "Juice:"]
                sections = {}
                for i, lab in enumerate(labels):
                    idx = diet.find(lab)
                    if idx != -1:
                        start = idx + len(lab)
                        # find next label position
                        next_idx = -1
                        for j in range(i+1, len(labels)):
                            k = diet.find(labels[j], start)
                            if k != -1:
                                next_idx = k
                                break
                        if next_idx == -1:
                            content = diet[start:].strip()
                        else:
                            content = diet[start:next_idx].strip()
                        sections[lab.rstrip(":")] = content

                if sections:
                    for key in ["Vegetables", "Protein Intake", "Juice"]:
                        val = sections.get(key)
                        if val:
                            text += f"\n{key}:\n{val}\n"
                else:
                    # fallback to full paragraph
                    text += f"\n{diet}"
            else:
                text += "\nNo diet listed"

            if isinstance(rec, str) and rec:
                text += f"\nNotes:\n{rec}"
            return text

    # Process user message
    def process(self, message):
        message = message.strip()

        # Step-by-step conversation
        if self.state["goal"] is None:
            self.state["goal"] = message
            return "Got it 👍 What is your gender? (Male/Female)"
        if self.state["gender"] is None:
            self.state["gender"] = message
            return "What is your weight in kg?"
        if self.state["weight"] is None:
            try:
                self.state["weight"] = float(message)
                return "What is your height in meters? (example: 1.65)"
            except ValueError:
                return "Please enter your weight as a number."
        if self.state["height"] is None:
            try:
                self.state["height"] = float(message)
                bmi_msg = self.calculate_bmi()
                return f"{bmi_msg}\nWhat is your age?"
            except ValueError:
                return "Please enter your height as a number (example: 1.65)."
        if self.state["age"] is None:
            try:
                self.state["age"] = int(message)
                return "Do you have Hypertension? (Yes/No)"
            except ValueError:
                return "Please enter your age as a number."
        if self.state["hypertension"] is None:
            self.state["hypertension"] = message
            return "Do you have Diabetes? (Yes/No)"
        if self.state["diabetes"] is None:
            self.state["diabetes"] = message
            return "What do you want? Exercise plan or Diet plan?"

        # If user chooses plan, ask for fitness type before providing plan
        if self.state["choice"] is None and ("exercise" in message.lower() or "diet" in message.lower()):
            # store normalized choice
            if "exercise" in message.lower():
                self.state["choice"] = "exercise"
            else:
                self.state["choice"] = "diet"
            # offer dataset-backed options if available
            try:
                types = ", ".join(self.df["Fitness Type"].unique())
                return f"Which fitness type are you interested in? Options: {types}"
            except Exception:
                return "Which fitness type are you interested in? (e.g., Cardio, Muscular)"

        # capture fitness type and then produce a dataset plan (with equipment/diet) and ask for recommendation
        if self.state["choice"] is not None and self.state["fitness_type"] is None:
            self.state["fitness_type"] = message
            plan_text = self.get_plan_from_dataset(self.state["choice"])
            # set awaiting_recommendation so next yes/no is interpreted
            self.state["awaiting_recommendation"] = self.state["choice"]
            return f"{plan_text}\n\nWould you like a personalized ML recommendation? (Yes/No)"

        # If we are awaiting a yes/no for recommendation
        if self.state.get("awaiting_recommendation") is not None:
            ans = message.strip().lower()
            choice = self.state.get("awaiting_recommendation")
            # clear awaiting flag
            self.state["awaiting_recommendation"] = None
            if ans in ["yes", "y"]:
                if choice == "exercise":
                    return self.ml_exercise_recommendation()
                else:
                    return self.ml_diet_recommendation()
            else:
                return "Okay — happy to help if you need anything else."

        # Provide ML recommendations
        if "exercise" in message.lower():
            return self.ml_exercise_recommendation()
        if "diet" in message.lower():
            return self.ml_diet_recommendation()

        return "Please choose 'Exercise plan' or 'Diet plan'."
