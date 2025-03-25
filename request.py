import requests

url = "http://localhost:8000/chat"
payload = {
    "messages": [
        {
            "role": "user",
            "content": "I've been having severe chest pain for the last hour"
        }
    ],
    "temperature": 0.0,
    "model": "llama-3.1-8b-instant",  # Updated to a Groq model
    "mode": "healthcare"  # Added mode specification
}

response = requests.post(url, json=payload)
print(response.json())