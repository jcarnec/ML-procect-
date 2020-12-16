import urllib
import jsonlines
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
    with open(filename) as f:
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
    print(full_dict["252950"]["name"])
    print(full_dict["252950"]["publisher"])
    print(full_dict["252950"]["positive"])
    print(full_dict["252950"]["genre"])


run()
