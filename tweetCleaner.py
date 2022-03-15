import TwitterAPICALL
import nltk
from nltk.corpus import stopwords
import numpy as np


class tweetCleaner():
    def __init__(self, jsonObj):
        self.json = jsonObj

    def removeLinksHashtags(self):


    def removeRepeats(self):
        repeatArray = np.array([])
        cleanedJson = np.array([])
        for item in self.json["data"]:
            if ('RT @' not in item["text"]):
            #if (('RT @' not in item["text"]) and (item["text"] not in repeatArray)):
                repeatArray = np.append(repeatArray, item["text"])
                cleanedJson = np.append(cleanedJson, item)

        self.cleanedJson = cleanedJson

    def removeStopWords(self):
        for item in self.json["data"]:
            print("hi")



json = TwitterAPICALL.callTwitter("Kanye Pete Davidson beef", 40)
g = tweetCleaner(json)
g.removeRepeats()
#print(g.cleanedJson)
for item in g.cleanedJson:
    print(item['text'])

#g.removeStopWords()

#like count, retweet count, comment count, follower count?