import os
import requests

BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

response = requests.get(f"{BASE_URL}/search", params={"q": "where do they talk about wellbeing resources"})
print(response.json())