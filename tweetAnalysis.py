import TwitterAPICALL
import tweetCleaner
from tweetCleaner import tweetCleaner
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib
from matplotlib import pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
import json
import modelAnalyzer
from modelAnalyzer import modelAnalyzer

def fit_tree(X, y, max_depth = 2, test_size = 0.25):
    T = tree.DecisionTreeClassifier(max_depth = max_depth)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    T.fit(X_train,y_train)
    return T, X_train, X_test, y_train, y_test

def best_fit_Tree(X, y):
    best_score = -np.inf
    scores = np.zeros(30)
    for d in range(1,31):
        T, X_train, X_test, y_train, y_test = fit_tree(X,y, max_depth = d)
        #print(y_train)
        scores[d-1] = cross_val_score(T, X_train, y_train).mean()
        if scores[d-1] > best_score:
            best_score = scores[d-1]
            best_depth = d
    T, X_train, X_test, y_train, y_test = fit_tree(X,y, max_depth = best_depth)
    print("Training score is: " + str(T.score(X_train, y_train)))
    print("Test score is: " + str(T.score(X_test, y_test)))
    print("best depth is: " + str(best_depth) )
    plt.scatter(np.arange(1,31),scores)
    fig, ax = plt.subplots(1, figsize = (20, 20))
    p = tree.plot_tree(T, filled = True, feature_names = X.columns)

def fit_SGDReg(X, y, test_size = 0.25):
    sgdr = SGDRegressor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    sgdr.fit(X_train,y_train)
    return sgdr, X_train, X_test, y_train, y_test

def fit_lasso(X, y, test_size = 0.25):
    lassoModel = Lasso()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    lassoModel.fit(X_train,y_train)
    return lassoModel, X_train, X_test, y_train, y_test

####
#myjson = TwitterAPICALL.getPastSevenDays("Bob Ross", 100)
#g = tweetCleaner(myjson)
#g.prepTweets()

#modelAnalysis = modelAnalyzer([])
#modelAnalysis.addJson(g.cleanedJson, "Bob_RossLikes")

#myjson = TwitterAPICALL.getPastSevenDays("Kanye Pete Davidson", 100)

#g = tweetCleaner(myjson)
#g.prepTweets()
#modelAnalysis.addJson(g.cleanedJson, "Kanye_PeteLikes")

#myjson = TwitterAPICALL.getPastSevenDays("Ukraine Russia", 100)

#g = tweetCleaner(myjson)
#g.prepTweets()
#modelAnalysis.addJson(g.cleanedJson, "Ukraine_RussiaLikes")
####

modelAnalysis = modelAnalyzer(['Bob_Ross.json','Kanye_Pete.json','Ukraine_Russia.json'])
modelAnalysis.fitAllData(minLikes = 10)

