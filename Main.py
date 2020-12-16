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


def assign_genre(filename, genre):
    with open(filename) as f:
        data = json.load(f)
    keys = []
    for key in data:
        keys.append(key)
    for key in keys:
        data[key]["genre"] = genre
    return data


action_data = assign_genre('Action_games.json', "Action")
adventure_data = assign_genre('Adventure_games.json', "Adventure")
early_access_data = assign_genre('Early_access_games.json', "Early Access")
ex_early_access_data = assign_genre('Ex_early_access_games.json', "Ex Early Access")
free_data = assign_genre('Free_games.json', "Free")
indie_data = assign_genre('Indie_games.json', "Indie")
rpg_data = assign_genre('Role_playing_games.json', "RPG")
simulation_data = assign_genre('Simulation_games.json', "Simulation")
strategy_data = assign_genre('Strategy_games.json', "Strategy")
sports_data = assign_genre('Sports_games.json', "Sports")

print(sports_data["44350"]["genre"])
