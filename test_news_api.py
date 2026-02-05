import requests

API_KEY = "3358307798f84164adf991cbb1990a6f"
url = "https://newsapi.org/v2/everything"

params = {
    "q": "united states economy",
    "apiKey": API_KEY,
    "language": "en",
    "pageSize": 3
}

response = requests.get(url, params=params)
print(response.status_code)
print(response.json())
