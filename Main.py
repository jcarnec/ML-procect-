import urllib
import numpy as np
import pandas
from sklearn import preprocessing
import requests
import json
import pickle as pk
from sklearn import neighbors
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
import math
import numpy as np
import pprint as pp
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC 
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

# Globals
spy_games = {}
products = {}



def load_scraped():

    output = 'scraper/steam-scraper/output/products_all.jl'

    with open(output) as f:
        for line in f:
            product = json.loads(line)
            try:
                products[product['id']] = product
            except KeyError:
                pass


def load_from_api():
    url = 'https://steamspy.com/api.php?request=genre&genre=Adventure'
    response = requests.get(url)
    print(response)
    url = "https://steamspy.com/api.php?request=all&page=1"


    response = urllib.urlopen(url)

    data = json.loads(response.read())

    return(data)




def assign_genre(filename, genre):
    with open(filename, encoding="utf8") as f:
        data = json.load(f)
    keys = []
    for key in data:
        keys.append(key)
    for key in keys:
        if key in spy_games:
            temp = make_list(genre[0])
            temp.extend(spy_games[key]["genre"])
            data[key]["genre"] = temp

        else:
            data[key]["genre"] = genre
    return data


def make_list(string):
    return [string]


def add_and_update_full_dict(filename, genre):
    genre_dict = assign_genre(filename, make_list(genre))
    spy_games.update(genre_dict)


def combine_files():
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


def get_mean_value(range_string):
    x = float(range_string.split(' ')[0].replace(',', ''))
    y = float(range_string.split(' ')[-1].replace(',', ''))
    return (y - x / 2)

def disqualify(game, product):
    invalid_publisher = (game["publisher"] == "" or game["developer"] == "") or not isinstance(game["publisher"], str)
    no_specs = ("specs" not in product.keys())
    return (invalid_publisher or no_specs)

def encode_mlb(info, key):
    mlb = MultiLabelBinarizer()
    df = pandas.DataFrame({"Type":info[key]})
    result = pandas.DataFrame(mlb.fit_transform(df['Type']),columns=mlb.classes_)
    return result.to_numpy()

def get_x_y():

    combine_files()

    info = {
        "genre": [],
        "publisher": [],
        "developer": [],
        "owners": [],
        "tags": [],
        "early_access": [],
        "specs": []
    }


    load_scraped()


    #  with open('products.pickle', 'wb+') as handle:
    #      pickle.dump(products, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #  with open('products.pickle', 'rb') as handle:
    #      products = pickle.load(handle)

    eliminated_count = 0
    for key in spy_games:
        if (key in products):
            spy_game = spy_games[key]
            product = products[key]
            if (disqualify(spy_game, product) == False):
                info["genre"].append(spy_game["genre"])
                info["publisher"].append(spy_game["publisher"])
                info["developer"].append(spy_game["developer"])
                info["owners"].append(get_mean_value(spy_game["owners"]))
                info["tags"].append(product['tags'])
                info["early_access"].append(int(product['early_access'] == True))
                info["specs"].append(product['specs'])

            else:
                eliminated_count += 1

    
    print('eliminated count', eliminated_count)

    info['genre'] = encode_mlb(info, 'genre')
    info['publisher'] = encode_mlb(info, 'publisher')
    info['developer'] = encode_mlb(info, 'developer')
    info['tags'] = encode_mlb(info, 'tags')
    info['specs'] = encode_mlb(info, 'specs')
    list_of_np_array = []
    for key in info:
        # if key == "tags" or key == "specs" or key == "genre" or key == "early_access":
        if key == "tags" or key == "genre":
            list_of_np_array.append(info[key])
    X = np.column_stack(list_of_np_array)
    Y = np.array(info['owners'])
    Y = Y / np.linalg.norm(Y) 
    return X, Y, info


X, Y, set = get_x_y()


kf = KFold(n_splits=3, shuffle=True)
for train, test in kf.split(X):
   model = LinearRegression().fit(X[train], Y[train])
   knn_model = KNeighborsRegressor(n_neighbors = 10).fit(X[train], Y[train])
   predictions = knn_model.predict(X[test])
   x_sorted = [x for _,x in sorted(zip(Y,predictions))]
   y_sorted = [y for y,_ in sorted(zip(Y,predictions))]
   print(y_sorted)
   print(len(y_sorted))
   plt.scatter(y_sorted, x_sorted)
   plt.xscale('log')
   plt.yscale('log')
   plt.show()



