""" -*- coding: utf-8 -*-"""
###############################################################################
#University:        Trinity College Dublin
#Course:            High Performance Computing (MSc)
#Module:            CS7CS4 Machine Learning
#Assignment:        Group Project 
#Lecturer:          Douglas Leith
#Author:            William O'Sullivan & Joseph Carnec(Group 43)
#Student No.:       16321101
#Contact email:     wosulliv@tcd.ie
#Created:           2020-12-15
#Due date:          2020-12-22
#IDE Used:          Spyder 4.1.5
#Version:           Python 3.7
###############################################################################
### Importing Libraries #######################################################
import numpy as np
import pandas as pd
from csv import reader
import json

import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor


#%% Joseph Carnec's code:
# Globals
spy_games = {}
products = {}

def load_scraped():
    output = 'scraper/steam-scraper/output/products_all.jl'
    with open(output, encoding="utf8") as f:
        for line in f:
            product = json.loads(line) 
            try:
                products[product['id']] = product
            except KeyError:
                pass

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


def get_price(product):
    if 'price' in product:
        price_string = product['price']
        if (type(price_string) == str):
            try:
                price_string = price_string.replace('€', '')
                num = int(price_string)
            except ValueError:
                return 0
        else:
            num = int(price_string)
        return num / 100
    else: return 0

def disqualify(game, product):
    no_specs = ("specs" not in product.keys())
    invalid_price = (get_price(product) == 0)
    return (no_specs or invalid_price)

def encode_mlb(info, key):
    mlb = MultiLabelBinarizer()
    df = pd.DataFrame({"Type":info[key]})
    result = pd.DataFrame(mlb.fit_transform(df['Type']),columns=mlb.classes_)
    return result.to_numpy()

def normalize(N): #for the n-th term in a list, N.
    #norm = lambda n, N: 2*((n-min(N))/(max(N)-min(N))) - 1 #normalises to values between -1 and 1
    norm = lambda n, N: ((n-min(N))/(max(N)-min(N))) #normalises to values between 0 and 1
    xnorm = [];
    for element in N:
        xnorm.append(norm(element,N))
    return np.array(xnorm)


def get_x_y():

    combine_files()



    with open('combined_table_target_features.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row, p in zip(csv_reader, spy_games):
            spy_games[p]['owners'] = row[0]


    info = {
        "genre": [],
        "publisher": [],
        "developer": [],
        "owners": [],
        "tags": [],
        "early_access": [],
        "specs": [],
        "price": []
    }


    load_scraped()

    eliminated_count = 0
    for key in spy_games:
        if (key in products):
            spy_game = spy_games[key]
            product = products[key]
            if (disqualify(spy_game, product) == False):
                info["genre"].append(spy_game["genre"])
                info["publisher"].append(spy_game["publisher"])
                info["developer"].append(spy_game["developer"])
                info["owners"].append(float(spy_game["owners"]))
                info["tags"].append(product['tags'])
                info["early_access"].append(int(product['early_access'] == True))
                info["specs"].append(product['specs'])
                info ["price"].append(get_price(product))
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
    Y = Y / np.linalg.norm(Y)  #np.array(normalize(Y)) #


    return X, Y, info


X, Y, set = get_x_y()
#%%  Define Baseline
xspace = np.linspace(0,600,len(Y))
ymeanline = np.mean(Y) + 0*xspace
ymse = mean_squared_error(ymeanline,Y)
ymseline = ymse + 0*xspace

#%% LASSO Regression - Tuning C hyperparameter
mean_list = []; var_list =  []
l = []; C = []
counter = 0

#Generating a set of alpha corresponding to C values found in the list C
for k in range(5,-11,-1):
    l.append(0.5**-k)        #alpha = 1/2C, C = 1/2alpha 
    C.append(1/(2*l[counter]))
    counter = counter + 1

#Testing the effects of varying alpha on accuracy of the data to the real model    
kf = KFold(n_splits=10, shuffle=True)
for a in l:
    mse_list = []
    for train,test in kf.split(X):
        model_lasso = Lasso(alpha=a, max_iter=10000).fit(X[train],Y[train])
        ypred_lasso = model_lasso.predict(X[test])
        mse_list.append(mean_squared_error(Y[test],ypred_lasso))
    mean_list.append(np.mean(mse_list))
    var_list.append(np.var(mse_list))
     
fig01 = plt.figure(figsize=(10,5))
ax01 = fig01.add_subplot(1,1,1)
Ctext01 = "Variation in Mean Squared Error with C for LASSO Regression"
ax01.set_title(Ctext01, fontweight="bold", fontsize=15)
ax01.set_xlabel('Hyperparameter ($C$)', fontweight="bold", fontsize=15)
ax01.set_ylabel('Mean Squared Error', fontweight="bold", fontsize=15)
ax01.set_yscale('log')
ax01.errorbar(C,mean_list,yerr=var_list,capsize=5, label='$\\bar{x}^2 \pm \sigma^2$' )
ax01.plot(xspace,ymseline, label='Baseline $\\bar{x}^2$ in y')
ax01.legend(fontsize=15, loc='best',fancybox= True,framealpha=0.97)
ax01.grid()

#%% Ridge Regression - Tuning C hyperparameter
mean_list = []; var_list =  []
l = []; C = []
counter = 0

#Generating a set of alpha corresponding to C values found in the list C
for k in range(5,-11,-1):
    l.append(0.5**-k)        #alpha = 1/2C, C = 1/2alpha 
    C.append(1/(2*l[counter]))
    counter = counter + 1

#Testing the effects of varying alpha on accuracy of the data to the real model    
kf = KFold(n_splits=10, shuffle=True)
for a in l:
    mse_list = []
    for train,test in kf.split(X):
        model_ridge = Ridge(alpha=a, max_iter=10000).fit(X[train],Y[train])
        ypred_ridge = model_ridge.predict(X[test])
        mse_list.append(mean_squared_error(Y[test],ypred_ridge))
    mean_list.append(np.mean(mse_list))
    var_list.append(np.var(mse_list))
     
fig02 = plt.figure(figsize=(10,5))
ax02 = fig02.add_subplot(1,1,1)
Ctext02 = "Variation in Mean Squared Error with C for Ridge Regression"
ax02.set_title(Ctext02, fontweight="bold", fontsize=15)
ax02.set_xlabel('Hyperparameter ($C$)', fontweight="bold", fontsize=15)
ax02.set_ylabel('Mean Squared Error', fontweight="bold", fontsize=15)
ax02.set_yscale('log')
ax02.errorbar(C,mean_list,yerr=var_list,capsize=5, label='$\\bar{x}^2 \pm \sigma^2$' )
ax02.plot(xspace,ymseline, label='Baseline $\\bar{x}^2$ in y')
ax02.legend(fontsize=15, loc='best',fancybox= True,framealpha=0.97)
ax02.grid()

#%% Redefining xspace for baseline for kNN graphing purposes:
xspace = np.linspace(0,60,len(Y))

#%% kNN Regression - Finding optimal k
mean_list = []; var_list =  []
gamma = 25
k_space = []

kf = KFold(n_splits=10, shuffle=True)
for k in range(2,50,1):
    mse_list = []
    k_space.append(k)
    for train,test in kf.split(X):
        model_knn = KNeighborsRegressor(n_neighbors=k,weights='distance').fit(X[train], Y[train])
        ypred_knn = model_knn.predict(X_poly[test])
        mse_list.append(mean_squared_error(Y[test],ypred_knn))
    mean_list.append(np.mean(mse_list))
    var_list.append(np.var(mse_list))

fig03 = plt.figure(figsize=(10,5))
ax03 = fig03.add_subplot(1,1,1)
Ctext03 = "Variation in Mean Squared Error with $k$ for kNN Regression, weighted by distance"
ax03.set_title(Ctext03, fontweight="bold", fontsize=13)
ax03.set_xlabel('Number of Nearest Neighbours ($k$)', fontweight="bold", fontsize=15)
ax03.set_ylabel('Mean Squared Error', fontweight="bold", fontsize=15)
ax03.set_yscale('log')
ax03.errorbar(k_space,mean_list, yerr=var_list,capsize=5, label='$\\bar{x}^2 \pm \sigma^2$' )
ax03.plot(xspace,ymseline, label='Baseline $\\bar{x}^2$ in y')
ax03.legend(fontsize=15, loc='best',fancybox= True,framealpha=0.97)
ax03.grid()

#%% Kernelised kNN Regression - Finding optimal k
def gaussian_kernel(distances):
    weights = np.exp(-(gamma) * (distances**2))
    return weights / np.sum(weights)

mean_list = []; var_list =  []
gamma = 25
k_space = []

kf = KFold(n_splits=10, shuffle=True)
for k in range(2,50,1):
    mse_list = []
    k_space.append(k)
    for train,test in kf.split(X):
        model_kknn = KNeighborsRegressor(n_neighbors=k,weights=gaussian_kernel).fit(X[train], Y[train])
        ypred_kknn = model_knn.predict(X[test])
        mse_list.append(mean_squared_error(Y[test],ypred_kknn))
    mean_list.append(np.mean(mse_list))
    var_list.append(np.var(mse_list))

fig04 = plt.figure(figsize=(10,5))
ax04 = fig03.add_subplot(1,1,1)
Ctext04 = "Variation in Mean Squared Error with $k$ for Kernelised kNN Regression"
ax04.set_title(Ctext03, fontweight="bold", fontsize=13)
ax04.set_xlabel('Number of Nearest Neighbours ($k$)', fontweight="bold", fontsize=15)
ax04.set_ylabel('Mean Squared Error', fontweight="bold", fontsize=15)
ax04.set_yscale('log')
ax04.errorbar(k_space,mean_list, yerr=var_list,capsize=5, label='$\\bar{x}^2 \pm \sigma^2$' )
ax04.plot(xspace,ymseline, label='Baseline $\\bar{x}^2$ in y')
ax04.legend(fontsize=15, loc='best',fancybox= True,framealpha=0.97)
ax04.grid()