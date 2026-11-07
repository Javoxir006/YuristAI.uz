import requests

url = "http://127.0.0.1:8000/v1/rag/batch"

# Spetsifikatsiyaga ko'ra Authorization header va Bearer kalit talab etiladi[cite: 2]
headers = {
    "Authorization": "Bearer test-api-key-123",
    "Content-Type": "application/json"
}

# Spetsifikatsiya formatidagi so'rov (Request)[cite: 2, 3]
payload = {
    "run_id": "eval-2026-11-03",
    "batch_id": "eval-2026-11-03-b01",
    "batch_index": 1,
    "total_batches": 10,
    "options": {
        "top_k": 5,
        "return_contexts": True,
        "return_debug": True,
        "temperature": 0,
        "seed": 42,
        "as_of_date": "2026-11-01",
        "answer_language": "auto"
    },
    "items": [
        {
            "id": "q001",
            "question": "Yillik asosiy mehnat ta'tilining eng kam muddati qancha?",
            "language": "uz-Latn"
        },
        {
            "id": "q002",
            "question": "Qonunda javobi yo'q savol namunasi",
            "language": "uz-Latn"
        }
    ]
}

response = requests.post(url, json=payload, headers=headers)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())