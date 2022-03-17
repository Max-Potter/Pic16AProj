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


myjson = TwitterAPICALL.getPastSevenDays("Bob Ross", 100)

g = tweetCleaner(myjson)
g.prepTweets()

#with open('json_data.json', 'w') as outfile:
#    json.dump(k.cleanedJson, outfile)

#with open('json_data.json') as json_file:
 #   data = json.load(json_file)

#g = tweetCleaner(data)
#g.prepTweets()

textList = []
likeCounts = []

for item in g.cleanedJson['data']:
    textList.append(item['text'])
    likeCounts.append(item['public_metrics']['like_count'])

data = {'text': textList, 'likeCounts': likeCounts}
df = pd.DataFrame(data)

nltk.download('punkt')
nltk.download('vader_lexicon')

tweets = df['text']
sid = SentimentIntensityAnalyzer()

scores = []
for tweet in tweets:
    score = sid.polarity_scores(tweet)
    scores.append(score['compound'])

df['sentiment_score']=scores

vec = CountVectorizer(stop_words = 'english')

counts = vec.fit_transform(df['text'])
count_df = pd.DataFrame(counts.toarray(), columns = vec.get_feature_names_out())
df = pd.concat((df,count_df),axis=1)

X = df.drop(['likeCounts','text'], axis=1)
y = df['likeCounts']

T, X_train, X_test, y_train, y_test = fit_tree(X, y, max_depth = 8)
print(T.score(X,y))
print(T.score(X_test,y_test))
print(T.score(X_train, y_train))
print("END ----------- END")
#best_fit_Tree(X,y)


#sgdr, X_train, X_test, y_train, y_test = fit_SGDReg(X,y)
#print(sgdr.score(X,y))
#print(sgdr.score(X_test,y_test))
#print(sgdr.score(X_train, y_train))

print("END ----------- END")

lasso, X_train, X_test, y_train, y_test = fit_lasso(X,y)
print(lasso.score(X,y))
print(lasso.score(X_test,y_test))
print(lasso.score(X_train, y_train))




# numTweetsWithWord = {"best" : 5, "worst" : 7}
# wordMatrixx -->    

#   (CurrentAvg * numTweetsWithThatWord (doesn't include currentTweet) )) + CurrentTweetsLikeCount
#    ----------------------------------------------------------
#               numTweetsWithThatWord + 1



# same matrix, we take every word it has a '1' or a '0' for each tweet as a row
# in each row of that same dataframe, we want an extra column that is the like count