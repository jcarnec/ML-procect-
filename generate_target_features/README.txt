#README

#Generates Target Features using the Owners value and the Positive and Negative Reviews.
#######################################################################################
#EXAMPLE:

#Import "generate_target_feature" class
from generate_target_feature import *

#Define normalisation for the output:
def normalise(N): #for the n-th term in a list, N.
    norm = lambda n, N: 2*((n-min(N))/(max(N)-min(N))) - 1 #normalises to values between -1 and 1
    xnorm = [];
    for element in N:
        xnorm.append(norm(element,N))
    return xnorm

#Call g_t_f with the filename
g1 = generate_target_feature('Strategy_games.json')
y = normalise(g1.output()) #Returns normalised output/ target feature
#######################################################################################

#NOTE: g1 takes ~180s to run

 