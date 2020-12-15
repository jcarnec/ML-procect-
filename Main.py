import urllib

import jsonlines
import requests
import json
import os

TEST_FILENAME = os.path.join(os.path.dirname(__file__), 'test.txt')
print(TEST_FILENAME)

products = []

output = 'scraper/steam-scraper/output/products_all.jl'

with jsonlines.open(output) as reader:
    for obj in reader:
        products.append(obj)

print(products[0])
'''
url = 'https://steamspy.com/api.php?request=all&page=1'
response = requests.get(url)
print(response)

url = "https://steamspy.com/api.php?request=all&page=1"

response = urllib.urlopen(url)

data = json.loads(response.read())

print(data)
'''

with open('games1.json') as f:
    data = json.load(f)

keys = []
for key in data:
    keys.append(key)

for key in keys:
    print(data[key]['owners'])
