import requests

class network_connections():
    def __init__(self) -> None:
        pass

    def send_post_to_llm(self, query, url_generate='http://127.0.0.1:5201/generate') -> dict:
        data = {"text": query}
        response = requests.post(url=url_generate, json=data)
        if response.status_code == 200:
            return response.json()['generated_text']
        else:
            print("Error:", response.status_code, response.text)
            if "is >= max_length" in response.text:
                return "chunk >= max_length"
            return None