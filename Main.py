import urllib
import jsonlines
import requests
import json


products = []

output = 'scraper/steam-scraper/output/products_all.jl'

with jsonlines.open(output) as reader:
    for obj in reader:
        products.append(obj)
        #data = obj['id']
        #print(data)

'''
url = 'https://steamspy.com/api.php?request=genre&genre=Adventure'
response = requests.get(url)
print(response)

url = "https://steamspy.com/api.php?request=all&page=1"

response = urllib.urlopen(url)

data = json.loads(response.read())

print(data)

'''
with open('Action_games.json') as f:
    data = json.load(f)

keys = []
for key in data:
    keys.append(key)

print(len(keys))

for key in keys:
    data[key]["genre"] = "Action"

print(data["359550"]["genre"])

