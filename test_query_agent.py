from utils.data_processing import load_dataset
from agents.query_agent import NaturalLanguageQueryAgent


# Step 1: Load sample dataset
data = load_dataset("sample.csv")

# Step 2: Initialize Query Agent
query_agent = NaturalLanguageQueryAgent()

# Step 3: Ask questions
questions = [
    "What is the average age?",
    "How many employees work in IT?",
    "Show the top 3 highest-paid employees.",
    "Which department has the highest average salary?"
]

# Step 4: Get answers
for question in questions:

    print("\n" + "=" * 60)
    print("QUESTION:", question)

    try:
        answer = query_agent.answer_question(question, data)

        print("AI INTERPRETED PLAN:")
        print(answer["plan"])

        print("RESULT:")
        print(answer["result"])

        print("AI EXPLANATION:")
        print(answer["explanation"])


    except Exception as error:
        print("ERROR:", error)


print("\n===== QUERY AGENT TEST COMPLETED =====")