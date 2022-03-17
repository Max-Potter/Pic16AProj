import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
import json
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import warnings
import copy
class modelAnalyzer():
    def __init__(self,jsonList):
        nltk.download('punkt')
        nltk.download('vader_lexicon')
        self.jsonList = jsonList
        self.prepData()

    def addJson(self,newJson, fileName):
        if len(fileName) < 5 or fileName[-5:] != ".json":
            fileName = fileName + ".json"
        self.jsonList.append(fileName)
        with open(fileName, 'w') as outfile:
            json.dump(newJson, outfile)

    def getAllJsons(self):
        allJsons = {}
        for jsonName in self.jsonList:
            with open(jsonName) as json_file:
                currJson = json.load(json_file)
            allJsons[jsonName] = currJson
        return allJsons

    def prepData(self, minLikes = 0):
        allJsons = self.getAllJsons()
        preppedData = []
        for dataKey in allJsons:
            dataFrame = allJsons[dataKey]
            textList = []
            likeCounts = []
            for item in dataFrame['data']:
                if item['public_metrics']['like_count'] > minLikes:
                    textList.append(item['text'])
                    likeCounts.append(item['public_metrics']['like_count'])
            data = {'text': textList, 'likeCounts': likeCounts}
            df = pd.DataFrame(data)
            tweets = df['text']
            sid = SentimentIntensityAnalyzer()
            scores = []
            for tweet in tweets:
                score = sid.polarity_scores(tweet)
                scores.append(score['compound'])
            df['sentiment_score']=scores
            vec = CountVectorizer(stop_words = 'english')
            counts = vec.fit_transform(df['text'])
            count_df = pd.DataFrame(counts.toarray(),columns = vec.get_feature_names_out())
            df = pd.concat((df,count_df),axis=1)
            X = df.drop(['likeCounts','text'],axis=1)
            y = df['likeCounts']
            preppedData.append([dataKey, X, y])
        self.preppedData = preppedData
        return preppedData

    def fit_tree(self, X, y, max_depth = 2, test_size = 0.25):
        T = tree.DecisionTreeClassifier(max_depth = max_depth)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
        T.fit(X_train,y_train)
        return T, X_train, X_test, y_train, y_test

    def best_fit_Tree(self, X, y):
        best_score = -np.inf
        scores = np.zeros(30)
        for d in range(1,31):
            T, X_train, X_test, y_train, y_test = self.fit_tree(X,y, max_depth = d)
            #print(y_train)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                scores[d-1] = cross_val_score(T, X_train, y_train).mean()
            if scores[d-1] > best_score:
                best_score = scores[d-1]
                best_depth = d
        return best_depth

    def fit_lasso(self, X, y, test_size = 0.25):
        lassoModel = Lasso()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
        lassoModel.fit(X_train,y_train)
        return lassoModel, X_train, X_test, y_train, y_test

    def fit_SGDReg(self, X, y, test_size = 0.25):
        sgdr = SGDRegressor()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
        sgdr.fit(X_train,y_train)
        return sgdr, X_train, X_test, y_train, y_test

    def fit_naiveBayes(self, inX,iny, test_size=0.25):
        X = copy.copy(inX)
        y = copy.copy(iny)
        X=X.drop(labels = 'sentiment_score', axis=1)
        #print(X)
        termFreq_Transformer=TfidfTransformer()
        X_termFreq = termFreq_Transformer.fit_transform(X)
        for index, val in y.iteritems():
            if val > 200:
                y[index] = "Viral"
            elif val > 65:
                y[index] = "Extremely Popular"
            elif val > 40:
                y[index] = "Very Popular"
            elif val > 25:
                y[index] = "Popular"
            elif val > 15:
                y[index] = "Slightly Popular"
            elif val > 10:
                y[index] = "Slightly Unpopular"
            elif val > 5:
                y[index] = "Very Unpopular"
            else:
                y[index] = "Extremely Unpopular"
        X_train, X_test, y_train, y_test = train_test_split(X_termFreq, y, test_size = test_size)
        bayesModel = MultinomialNB().fit(X_train,y_train)
        return bayesModel, X_train, X_test, y_train, y_test

    def fitAllData(self, minLikes = 0):
        allData = self.prepData(minLikes = minLikes)
        for dataSet in allData:
            name = dataSet[0]
            X = dataSet[1]
            y = dataSet[2]
            bestTreeDepth = self.best_fit_Tree(X,y)
            T, X_train, X_test, y_train, y_test = self.fit_tree(X, y, max_depth = bestTreeDepth)
            print("Scores for dataset " + str(name))
            print(" --------- ")
            print("Decision Tree Score on Training Data: " + str(T.score(X_train,y_train)))
            print("Decision Tree Score on Test Data: " + str(T.score(X_test,y_test)))
            print(" --------- ")
            lasso, X_train, X_test, y_train, y_test = self.fit_lasso(X,y)
            print("Lasso Score on Training Data: " + str(lasso.score(X_train,y_train)))
            print("Lasso Score on Test Data: " + str(lasso.score(X_test,y_test)))
            print(" --------- ")
            bayesModel, X_train, X_test, y_train, y_test = self.fit_naiveBayes(X,y)
            print("Baye's Model Score on Training Data: " + str(bayesModel.score(X_train,y_train)))
            print("Baye's Model Score on Test Data: " + str(bayesModel.score(X_test,y_test)))
            print(" /////////////// ")
           






    

            

        


