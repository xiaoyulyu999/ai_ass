import requests
import json
import pandas as pd

# Path to your CSV file
PATH = "./data/salaries_2023.csv"

# Read and clean the CSV
db = pd.read_csv(PATH)

db.fillna(0, inplace=True)

# Convert part of the DataFrame into a text context for the model
# (You can adjust how much to send — entire CSV may be too large)
data_sample = db.head(20).to_string(index=False)  # sample 20 rows
context = f"Here is some data from the salaries dataset:\n{data_sample}\n\n"

# Ask a question about the data
user_question = input("Ask a question about the salary data: ")

# Combine context + question
prompt = f"{context}\nQuestion: {user_question}\n\nAnswer based only on the data above."

# Setup the Ollama API URL
url = "http://localhost:11434/api/chat"

# Define the payload
payload = {
    "model": "gemma3:1b",
    "messages": [
        {"role": "system", "content": "You are a data assistant that answers questions using the provided CSV data."},
        {"role": "user", "content": prompt}
    ]
}

# Send POST request with streaming enabled
response = requests.post(url, json=payload, stream=True)

if response.status_code == 200:
    print("\nResponse from Ollama:\n")
    for line in response.iter_lines(decode_unicode=True):
        if line:
            try:
                json_data = json.loads(line)
                if "message" in json_data and "content" in json_data["message"]:
                    print(json_data["message"]["content"], end="")
            except json.JSONDecodeError:
                print(f"\nFailed to parse line: {line}")
    print()
else:
    print(f"Error: {response.status_code}")
    print(response.text)

## give memory context abilities