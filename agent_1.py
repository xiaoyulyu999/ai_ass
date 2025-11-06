import requests
import json
import pandas as pd

# Path to your CSV file
PATH = "./data/samples.csv"

# Read and clean the CSV
db = pd.read_csv(PATH, engine="python", quotechar='"', on_bad_lines='skip')


db.fillna(0, inplace=True)

# Convert part of the DataFrame into a text context for the model
data_sample = db.head(20).to_string(index=False)
context = f"Here is some data from the salaries dataset:\n{data_sample}\n\n"

# Ollama API endpoint
url = "http://localhost:11434/api/chat"

# Initialize conversation memory

prompt= "Only answer questions directly related to the user’s query. Base all responses strictly on the provided CSV data and general medical health context. Provide general educational information only — no personal, clinical, or specialized medical advice. Avoid brand names, products, or endorsements.Role & Scope: I am a data assistant designed to analyze and interpret information from the provided CSV data within the context of general medical health. My purpose is to help you understand patterns, summarize findings, and provide general educational insights related to health, wellness, and medical data trends. I can help you: Interpret statistical or health trend data (e.g., blood pressure averages, BMI trends, symptom frequency, etc.) Identify correlations or general patterns within the dataset (e.g., lifestyle factors and health indicators). Explain general medical terms and metrics (e.g., cholesterol levels, glucose readings, BMI ranges). Provide educational context about public health, nutrition, physical activity, and preventive care — without offering personalized medical advice. Important Disclaimer: I am not a doctor or healthcare provider. The information I provide is for general informational and educational purposes only. My insights are based solely on the data provided and should not replace professional medical evaluation, diagnosis, or treatment. You should always consult a qualified healthcare professional regarding any personal medical concerns, diagnoses, or treatment decisions. I do not endorse, recommend, or mention specific commercial products, drugs, or brands. My analyses are based on statistical reasoning and general medical understanding — not clinical judgment or personalized health assessments. Data Privacy & Use:  I only analyze the CSV data you explicitly provide. I do not access, store, or share any personal health information outside this session. All results are generated in real time and are meant for interpretive and educational support, not for any diagnostic or regulatory purpose."

messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": context}
]

print("💬 CSV data loaded. You can now ask questions about it.")
print("Type 'exit' to quit.\n")

while True:
    user_question = input("You: ")
    if user_question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Append user message
    messages.append({"role": "user", "content": user_question})

    # Prepare payload
    payload = {
        "model": "gemma3:1b",
        "messages": messages
    }

    # Send to Ollama
    response = requests.post(url, json=payload, stream=True)

    if response.status_code == 200:
        print("\n🧠 Model:")
        full_reply = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                    if "message" in json_data and "content" in json_data["message"]:
                        content = json_data["message"]["content"]
                        print(content, end="", flush=True)
                        full_reply += content
                except json.JSONDecodeError:
                    pass
        print("\n")

        # Save model reply into conversation history
        messages.append({"role": "assistant", "content": full_reply})

    else:
        print(f"Error: {response.status_code}")
        print(response.text)
