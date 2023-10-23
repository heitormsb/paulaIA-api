import requests

# url = "http://localhost:8000/api/v1/ia1"  # Update with your endpoint URL
#url =  'https://paulaia-api-joffihckwq-uc.a.run.app/api/v1/ia1'
url = "http://localhost:8000/api/v1/ia1"  # Update with your endpoint URL
file_path = "a.png"  # Path to the audio file you want to upload
# file_path = "move.png"  # Path to the audio file you want to upload
# file_path = "casa.png"  # Path to the audio file you want to upload

with open(file_path, "rb") as file:
    files = {"file": file}

    response = requests.post(url, files=files)

# with open(file_path, "rb") as f:
#     r = requests.post(url, files={'file': f})
#     print(r.json())

print(response.json())