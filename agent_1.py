import requests
import json
import pandas as pd

# Path to your CSV file
PATH = "./data/salaries_2023.csv"

# Read and clean the CSV
db = pd.read_csv(PATH)
db.fillna(0, inplace=True)

# Convert part of the DataFrame into a text context for the model
data_sample = db.head(20).to_string(index=False)
context = f"Here is some data from the salaries dataset:\n{data_sample}\n\n"

# Ollama API endpoint
url = "http://localhost:11434/api/chat"

# Initialize conversation memory
messages = [
    {"role": "system", "content": "You are a data assistant that answers questions using the provided CSV data."},
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
