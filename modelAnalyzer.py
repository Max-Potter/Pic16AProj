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


#modelAnalyzer is a class intended to load, store, and most importantly analyze tweets.
#It comes with 4 model types available to fit twitter data to, with all models
#using NLTK sentiment analysis values and the 'bag of words' count_vectorizer from scikit as predictor variables,
#and the number of likes on the tweet as the target variable.
#These 4 models are: Decision Trees, Lasso Models, SGDR Models, and MultinomialNB models
#modelAnalyzer must be initialized with a list 'jsonList' of all jsons we want to load,
#although this list can be empty and later updated using updateJsonList() and addJson()
#Member variables:
#self.jsonList: A list of strings, where each string is the name of a json object to be loaded
#self.preppedData: A list containing each json object, following modifications to make the data fit for scikit models
class modelAnalyzer():
    #Initializor. Automatically downloads nltk files if not already up to date.
    #Stores the list of jsons in self.jsonList
    #Automatically calls prepData() to prepare and format data for data analysis.
    #@params
    #jsonList: A list of strings, where each string is the name of a json object to be loaded. Can be empty.
    def __init__(self,jsonList):
        nltk.download('vader_lexicon')      #vader is a model for text sentiment analysis that's sensitive to both 
        nltk.download('punkt')              #polarity (positive/negative) and intensity of emotion and can understand 
                                            #the basic context of these words
        self.jsonList = jsonList
        self.preppedData = self.prepData()

        
    #Updates the jsonList to include indicated filename. 
    #Appends .json to the filename if not provided.
    #@params
    #fileName: A string. Intended name of file to be added. 
    #Returns the updated fileName. 
    def updateJsonList(self, fileName):
        #Check if file includes '.json' at the end. Checks if len(fileName) < 5 first to avoid index errors
        if len(fileName) < 5 or fileName[-5:] != ".json":
            fileName = fileName + ".json"
        #Ensures file does not already exist
        if fileName not in self.jsonList:
            self.jsonList.append(fileName)
        return fileName

    
    #Saves inputted json object into the current directory, and adds it to the JsonList
    #@params
    #newJson: A dictionary. Valid json objects are those returned from the TwitterAPICALL api call functions. 
    #fileName: String. Intended filename of the json object being saved
    def addJson(self,newJson, fileName):
        #Adds filename to list, and obtains proper .json name from updateJsonList
        newName = self.updateJsonList(fileName)
        with open(newName, 'w') as outfile:
            #Saves to current directory.
            json.dump(newJson, outfile)
            print("Successfully added " + newName + " to modelAnalyzer")
            

    #Creates a dictionary, where the keys are the json filenames and the values are the corresponding json objects
    #Raises an exception if the jsonList contains files that do not exist in the directory.
    #Returns the dictionary of all Jsons.
    def getAllJsons(self):
        allJsons = {}
        for jsonName in self.jsonList:
            #Exception Handling -- Ensures file exists first
            try:
                f = open(jsonName)
                f.close()
            except:
                raise Exception("ERROR -- " + str(jsonName) + " does not exist. Please add this file to the current directory.")
            with open(jsonName) as json_file:
                #Opens and loads the json object
                currJson = json.load(json_file)
            #Add json object to dictionary
            allJsons[jsonName] = currJson
        return allJsons

    
    #Prepares the data for further analysis. Prepped data is returned.
    #Note: All dataframes (json objects) are kept separate, and returned in a list as preppedData. 
    #To prepare data, a dataframe is created with the following columns:
    #'text': the tweet text
    #'likeCounts' : number of likes tweet received
    #'sentiment_score' : overall score from NTLK sentiment analyzer
    #text additionally undergoes 'bag of words' count vectorization, which make up all the remaining columns of dataframe.
    #@kwargs
    #minLikes: An integer. Any tweets that do not have more likes than minLikes are excluded from prepped data.
    def prepData(self, minLikes = 0):
        #Loads all Json objects
        allJsons = self.getAllJsons()
        #Creates a list of the prepped dataframes
        preppedData = []
        #Iterates through all json objects
        for dataKey in allJsons:
            dataFrame = allJsons[dataKey]
            #Construct a dataframe including the text of tweets and their number of likes
            textList = []
            likeCounts = []
            #Iterates through each tweet in the retrieved json
            for item in dataFrame['data']:
                #As long as the tweet has enough likes, it is added to the dataframe
                if item['public_metrics']['like_count'] > minLikes:
                    textList.append(item['text'])
                    likeCounts.append(item['public_metrics']['like_count'])
            data = {'text': textList, 'likeCounts': likeCounts}
            #converts collected data into a pandas dataframe
            df = pd.DataFrame(data)
            tweets = df['text']
            #NLTK Sentiment Analysis
            sid = SentimentIntensityAnalyzer()
            scores = []
            #Calculate overall sentiment score for every tweet, and add it to the dataframe
            for tweet in tweets:
                score = sid.polarity_scores(tweet)
                scores.append(score['compound'])
            df['sentiment_score']=scores

            #'bag of words' vectorization of the tweets -- creates large matrix of occurences of every word.
            vec = CountVectorizer(stop_words = 'english')
            counts = vec.fit_transform(df['text'])
            #Depending on version of sklearn, get_feature_names_out() may not work. This handles that exception
            try:
                count_df = pd.DataFrame(counts.toarray(),columns = vec.get_feature_names_out())
            except:
                count_df = pd.DataFrame(counts.toarray(),columns = vec.get_feature_names())
            #Adds the count vectorization to the existing dataframe
            df = pd.concat((df,count_df),axis=1)
            #Separates predictor variables (all but likecounts and text) from target variable(likecounts)
            X = df.drop(['likeCounts','text'],axis=1)
            y = df['likeCounts']
            #Adds newly prepped data to dataframe
            preppedData.append([dataKey, X, y])
        #self.preppedData = preppedData
        return preppedData


    #Converts the likecounts into categories of Extremely Unpopular up to Extremely Popular / Viral
    #@params
    #iny -- The 'y' dataframe, as created by preppedData. Dataframe must have one column of integers. 
    #      Dataframe is copied, so original data is preserved
    #Returns newly categorized dataframe
    def categorizeLikes(self, iny):
        #copies the dataframe
        y = copy.copy(iny)
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
        return y


    #Fits a Decision Tree to the inputted data
    #@params
    #X: Predictor Variables (multi-column dataframe including sentiment and bag of words count vectorizer)
    #iny: Target Variable (single column dataframe)
    #@kwargs
    #max_depth: max_depth of tree, default to 2
    #test_size: test_size used for train_test_split(), default to 0.25
    #Returns the tree, X_train set, X_test set, y_train set, and y_test set
    def fit_tree(self, X, iny, max_depth = 2, test_size = 0.25):
        #Convert likes into categorical data
        y = self.categorizeLikes(iny)
        T = tree.DecisionTreeClassifier(max_depth = max_depth)
        #Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
        T.fit(X_train,y_train)
        return T, X_train, X_test, y_train, y_test

    
    #Finds the tree depth that produces the highest accuracy for inputted data.
    #Returns the best depth, as well as a figure plotting the accuracy for all tested depths
    #@params
    #X: Predictor Variables (multi-column dataframe including sentiment and bag of words count vectorizer)
    #iny: Target Variable (single column dataframe)
    #@kwargs
    #max_tree_depth: max_depth of tree, default to 2
    def best_fit_Tree(self, X, iny, max_tree_depth = 30):
        #Initialize best score to -infinity
        best_score = -np.inf
        scores = np.zeros(30)
        y = copy.copy(iny)
        #Initialize plot of all depth's performances
        fig, ax = plt.subplots(1, figsize = (10,10))
        plotX = []
        plotY = []
        ax.set(xlabel = "Depth", ylabel = "Score", title = "Best Decision Tree Depth")
        #Iterates through all depths from 1 -> max_tree_depth
        for d in range(1,max_tree_depth + 1):
            #Fits a tree with specific max_depth
            T, X_train, X_test, y_train, y_test = self.fit_tree(X,y, max_depth = d)
            #print(y_train)
            #Computes cross-validation-score. Note -- Sometimes, the api call will retrieve tweets where
            #one category of likes does not have more than 5 entries. This causes repeated warning messages, so it is suppressed here.
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                scores[d-1] = cross_val_score(T, X_train, y_train).mean()
            #Updates best score
            if scores[d-1] > best_score:
                best_score = scores[d-1]
                best_depth = d
            #Plots depth against score
            plotX.append(d)
            plotY.append(scores[d-1])
        ax.plot(plotX,plotY)
        return best_depth, fig

    
    #Fits a lasso model to the inputted data
    #@params
    #X: Predictor Variables (multi-column dataframe including sentiment and bag of words count vectorizer)
    #y: Target Variable (single column dataframe)
    #@kwargs
    #test_size: test_size used for train_test_split(), default to 0.25
    #Returns the lasso model, X_train set, X_test set, y_train set, and y_test set
    def fit_lasso(self, X, y, test_size = 0.25):
        lassoModel = Lasso()
        #Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
        lassoModel.fit(X_train,y_train)
        return lassoModel, X_train, X_test, y_train, y_test


    #Fits a SGD Regression model to the inputted data
    #@params
    #X: Predictor Variables (multi-column dataframe including sentiment and bag of words count vectorizer)
    #y: Target Variable (single column dataframe)
    #@kwargs
    #test_size: test_size used for train_test_split(), default to 0.25
    #Returns the SGDR model, X_train set, X_test set, y_train set, and y_test set
    #Note that currently, some datasets cause this function to run for an extreme
    #amount of time, and as such it is not used anywhere in the project. 
    def fit_SGDReg(self, X, y, test_size = 0.25):
        sgdr = SGDRegressor()
        #Splits data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
        sgdr.fit(X_train,y_train)
        return sgdr, X_train, X_test, y_train, y_test

    #Fits a naive Bayes (MultinomialNB) model to the inputted data
    #Converts inputted iny variable to categorical data before fitting model. Does not use sentiment score.
    #@params
    #inX: Predictor Variables (multi-column dataframe including sentiment and bag of words count vectorizer)
    #iny: Target Variable (single column dataframe)
    #@kwargs
    #test_size: test_size used for train_test_split(), default to 0.25
    #Returns the naive Bayes model, X_train set, X_test set, y_train set, and y_test set
    def fit_naiveBayes(self, inX,iny, test_size=0.25):
        X = copy.copy(inX)
        #Convert y to categorized data
        y = self.categorizeLikes(iny)
        #Removes sentiment score from analysis
        X=X.drop(labels = 'sentiment_score', axis=1)

        #To account for differences in tweet length, the TfidfTransformer is used to convert counts into 'term frequencies'
        #This generally improves Naive Bayes model performance on larger text documents, but may very well be negligible for short tweets.
        termFreq_Transformer=TfidfTransformer()
        X_termFreq = termFreq_Transformer.fit_transform(X)
        #Splits data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_termFreq, y, test_size = test_size)
        bayesModel = MultinomialNB().fit(X_train,y_train)
        return bayesModel, X_train, X_test, y_train, y_test


    #Iterates through all data currently loaded into the modelAnalyzer, and prints out 
    #scores from each of 3 types of models (Decision Tree, Lasso, Naive Bayes) for each dataset.
    #Constructs a bar chart showing the scores of each model for each dataset, and returns it.
    #Note that models performing worse than -1 are automatically adjusted to a score of -1, to improve graph visibility.
    #We should never want to use a model with a score of -1 or less, anyways. 
    #@params
    #minLikes: Minimum number of likes a tweet must have received to be used
    #max_tree_Depth: Maximum tree depth for the Decision Tree
    def fitAllData(self, minLikes = 0, max_tree_depth = 30):
        #Loads all data
        allData = self.prepData(minLikes = minLikes)

        #initial set-up of bar chart
        xTicks = np.arange(len(allData))
        width = 0.3
        fig, ax = plt.subplots()
        ax.set(ylabel = 'Scores', title="Best Model for Dataset")
        nameList = []

        #Iterates through each dataSet currently loaded
        for count, dataSet in enumerate(allData):
            #Unpack data from each dataSet
            name = dataSet[0]
            nameList.append(name)
            X = dataSet[1]
            y = dataSet[2]
            #Find bestTreeDepth for the current dataset
            bestTreeDepth, tempFig = self.best_fit_Tree(X,y, max_tree_depth = max_tree_depth)
            plt.close(tempFig)
            #Fits a tree to best tree depth, and reports the scores.
            T, X_train, X_test, y_train, y_test = self.fit_tree(X, y, max_depth = bestTreeDepth)
            print("Scores for dataset " + str(name))
            print(" --------- ")
            print("Best Depth for Decision Tree: " + str(bestTreeDepth))
            trainScore = T.score(X_train,y_train)
            testScore = T.score(X_test,y_test)
            print("Decision Tree Score on Training Data: " + str(trainScore))
            print("Decision Tree Score on Test Data: " + str(testScore))
            #Adjusts score if it is too negative, to keep bar chart readable
            if testScore < -1:
                print("Adjusting for plot, score too negative")
                testScore = -1
            #Plot the test score of the model
            ax.bar(count - width, testScore, width, color = "green")
            print(" --------- ")
            lasso, X_train, X_test, y_train, y_test = self.fit_lasso(X,y)
            trainScore = lasso.score(X_train,y_train)
            testScore = lasso.score(X_test,y_test)
            print("Lasso Score on Training Data: " + str(trainScore))
            print("Lasso Score on Test Data: " + str(testScore))
            #Adjusts score if it is too negative, to keep bar chart readable
            if testScore < -1:
                print("Adjusting for plot, score too negative")
                testScore = -1
            #Plot the test score of the model
            ax.bar(count, testScore, width, color = "blue")
            print(" --------- ")
            bayesModel, X_train, X_test, y_train, y_test = self.fit_naiveBayes(X,y)
            trainScore = bayesModel.score(X_train,y_train)
            testScore = bayesModel.score(X_test,y_test)
            print("Bayes Model Score on Training Data: " + str(trainScore))
            print("Bayes Model Score on Test Data: " + str(testScore))
            #Adjusts score if it is too negative, to keep bar chart readable
            if testScore < -1:
                print("Adjusting for plot, score too negative")
                testScore = -1
            #Plot the test score of the model
            ax.bar(count + width, testScore, width, color = "orange")
            print(" /////////////// ")
        #Attempts to label barchart by each dataset. This fails on Matplotlib versions older than 3.5, so this lets the code continue running
        #but will result in an unlabeled bar chart being returned. 
        try:
            ax.set_xticks(xTicks, nameList)
        except:
            print("Warning: You are using an older version of MatPlotLib. Please upgrade to version 3.5 or newer to ensure proper display of graphs. Bar Graphs may not display categories properly otherwise.")
        #Adds a horizontal line at y = 0
        plt.axhline(y = 0.0, color = 'red', linestyle = "-")
        ax.legend(labels = ["X-axis","Decision Tree", "Lasso", "Bayes Model"])
        fig.tight_layout()
        return fig

    #Returns a dictionary of all data, where the key is the name of the dataSet and the values are a tuple containing the data (X,y) for analysis
    #@params
    #minLikes: Minimum number of likes a tweet must have to be included
    def getAllData(self, minLikes = 0):
        #Gets a list of all dataframe names, X, and y variables
        allData = self.prepData(minLikes = minLikes)
        dataDict = {}
        #Converts this list to a dictionary, indexed by dataframe name for the X,y values.
        for dataSet in allData:
            dataDict[dataSet[0]] = (dataSet[1],dataSet[2])
        return dataDict






    

            

        


