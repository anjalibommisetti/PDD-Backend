import requests
files = {'file': open('c:/App/sample.jpg', 'rb')}
resp = requests.post('http://127.0.0.1:8000/predict', files=files)
print('Status:', resp.status_code)
print('JSON:', resp.json())
