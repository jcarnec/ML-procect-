# -*- coding: utf-8 -*-
###############################################################################
#University:        Trinity College Dublin
#Course:            High Performance Computing (MSc)
#Module:            CS7CS4 Machine Learning
#Assignment:        Group Project - Output Designer Class
#Lecturer:          Douglas Leith
#Author:            William O'Sullivan (Group 43)
#Student No.:       16321101
#Contact email:     wosulliv@tcd.ie
#Created:           2020-12-18
#Due date:          2020-12-22
#IDE Used:          Spyder 4.1.5
#Version:           Python 3.7
###############################################################################
class generate_target_feature:
    def __init__(self,filename):
        import pandas as pd
        import numpy as np
        self.filename = 'Strategy_games.json'
        df = pd.read_json(self.filename,orient='index')         #Scan in .json file
        total_reviews = np.array(df.iloc[:,5] + df.iloc[:,6])   #Sum negative and positive reviews
        owners_range = np.array(df.iloc[:,8])                   #Obtain string values for ranges of ownership
        owners_mean = np.empty_like(owners_range)               #Allocate an array for the int averages of the range
        mean_list = [10000,35000,75000,150000,350000,750000,1500000,3500000,7500000,15000000,150000000]
        
        #Assigns average int values of the ranges to owners_mean in corresponding entries
        for i in range(0,len(owners_range)):
            if owners_range[i] == '0 .. 20,000':
                owners_mean[i] = mean_list[0]
            elif owners_range[i] == '20,000 .. 50,000':
                owners_mean[i] = mean_list[1]
            elif owners_range[i] == '50,000 .. 100,000':
                owners_mean[i] = mean_list[2]
            elif owners_range[i] == '100,000 .. 200,000':
                owners_mean[i] = mean_list[3]
            elif owners_range[i] == '200,000 .. 500,000':
                owners_mean[i] = mean_list[4]
            elif owners_range[i] == '500,000 .. 1,000,000':
                owners_mean[i] = mean_list[5]
            elif owners_range[i] == '1,000,000 .. 2,000,000':
                owners_mean[i] = mean_list[6]
            elif owners_range[i] == '2,000,000 .. 5,000,000':
                owners_mean[i] = mean_list[7]
            elif owners_range[i] == '5,000,000 .. 10,000,000':
                owners_mean[i] = mean_list[8]
            elif owners_range[i] == '10,000,000 .. 20,000,000':
                owners_mean[i] = mean_list[9]
            elif owners_range[i] == '100,000,000 .. 200,000,000':
                owners_mean[i] = mean_list[10]
                
        matrix = np.zeros((len(mean_list),len(owners_mean)))
        norm_matrix = np.zeros((len(mean_list),len(owners_mean)))
        
        for i in range(0,len(mean_list)):
                for j in range(0,len(owners_mean)):
                    if owners_mean[j] == mean_list[i]:
                        matrix[i,j] = total_reviews[j]
                norm_matrix[i,:] = self.normalise(matrix[i])
            
        scaling_params = np.zeros_like(owners_mean)
        self.num_owners = np.zeros_like(owners_mean)
        for i in range(0,len(mean_list)):
            for j in range(0,len(owners_mean)):
                if owners_mean[j] == mean_list[i]:
                    scaling_params[j] = norm_matrix[i,j]
                    self.num_owners[j] = (mean_list[i]*scaling_params[j])+1
                    self.num_owners[j] = int(np.round(self.num_owners[j]))
        
                
    def normalise(self,N): #normalises for the n-th term in a list, N.
        #norm = lambda n, N: 2*((n-min(N))/(max(N)-min(N))) - 1     #normalises to values between -1 and 1
        norm = lambda n, N: ((n-min(N))/(max(N)-min(N)))            #normalises to values between 0 and 1
        xnorm = [];
        for element in N:
            xnorm.append(norm(element,N))
        return xnorm
    
    def output(self):
        return self.num_owners