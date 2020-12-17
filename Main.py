import urllib
import jsonlines
import numpy as np
from sklearn import preprocessing
import requests
import json

'''
products = []

output = 'scraper/steam-scraper/output/products_all.jl'

with jsonlines.open(output) as reader:
    for obj in reader:
        products.append(obj)
        #data = obj['id']
        #print(data)
url = 'https://steamspy.com/api.php?request=genre&genre=Adventure'
response = requests.get(url)
print(response)

url = "https://steamspy.com/api.php?request=all&page=1"

response = urllib.urlopen(url)

data = json.loads(response.read())

print(data)

'''
full_dict = {}


def assign_genre(filename, genre):
    with open(filename, encoding="utf8") as f:
        data = json.load(f)
    keys = []
    for key in data:
        keys.append(key)
    for key in keys:
        if key in full_dict:
            temp = make_list(genre[0])
            temp.extend(full_dict[key]["genre"])
            data[key]["genre"] = temp

        else:
            data[key]["genre"] = genre
    return data


def make_list(string):
    return [string]


def add_and_update_full_dict(filename, genre):
    genre_dict = assign_genre(filename, make_list(genre))
    full_dict.update(genre_dict)


def run():
    add_and_update_full_dict('Action_games.json', "Action")
    add_and_update_full_dict('Adventure_games.json', "Adventure")
    add_and_update_full_dict('Early_access_games.json', "Early Access")
    add_and_update_full_dict('Ex_early_access_games.json', "Ex Early Access")
    add_and_update_full_dict('Free_games.json', "Free")
    add_and_update_full_dict('Indie_games.json', "Indie")
    add_and_update_full_dict('Role_playing_games.json', "RPG")
    add_and_update_full_dict('Simulation_games.json', "Simulation")
    add_and_update_full_dict('Strategy_games.json', "Strategy")
    add_and_update_full_dict('Sports_games.json', "Sports")


def normalise(z):
    return (z - min(z)) / (max(z) - min(z))


run()
all_keys = []
for key in full_dict:
    all_keys.append(key)

X1 = []  # parameter 1
for key in all_keys:
    X1.append(int(full_dict[key]["positive"]))

X1 = np.array(X1)
X1 = X1.reshape(-1, 1)
normalised_X1 = normalise(X1)  # preprocessing.normalize() function does not seem to be working correctly
print(normalised_X1)

y = []  # output
for key in all_keys:
    y.append(full_dict[key]["owners"])

print(len(y))
