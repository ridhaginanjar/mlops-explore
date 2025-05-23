import requests

image_path = './image_test/PNEUMONIA.jpeg'
url = 'http://localhost:8000/predict'

with open(image_path, 'rb') as image:
    file = {"file": image}
    response = requests.post(url, files=file)

# Cetak hasil respons
print("Status code:", response.status_code)
print("Response:", response.json())