import TwitterAPICALL
import nltk
from nltk.corpus import stopwords
import numpy as np
import copy

#tweetCleaner is class intended to clean up the raw json returned by twitter API Calls into something usable for data analysis
#All tweets should be cleaned by tweetCleaner before being passed on to a modelAnalyzer, although it is not necessary for the modelAnalyzer to work.
#calling self.prepTweets() will perform all necessary functions.
#tweetCleaner must be initialized with a jsonObj. Valid objects are those returned from api calls from TwitterAPICALL.py
class tweetCleaner():
    #Initializor. 
    #Initializes self.json and self.cleanedJson to the inputted json object.
    #self.json will not be changed after initialization, but self.cleanedJson will get updated with each cleaning function called
    #@params
    #jsonObj: json object (dictionary). Should be the json returned from a twitter api call. 
    def __init__(self, jsonObj):
        self.json = jsonObj
        self.cleanedJson = copy.copy(jsonObj)

    #Lowers text of all tweets to lowercase
    def lowerTweets(self):
        for item in self.cleanedJson['data']:
            item['text'] = item['text'].lower()

    #Removes links from twitter text
    def removeLinks(self):
        #initialize a rawstring for endstops
        endStop = r"\n"
        #Iterates through each tweet
        for item in self.cleanedJson['data']:
            tweetText = item['text']
            textArr = tweetText.split(' ')
            textCopy = copy.copy(textArr)
            #Goes through each word in the tweet's text, split by spaces.
            #If the word contains 'http' (eg: is a link), it is removed entirely
            for word in textCopy:
                if 'http' in word:
                    textArr.remove(word)
            #Rejoins the text, without any links, into one string.
            newText = ' '.join(textArr)
            #Removes endstops, as some texts have raw endstops in them
            newText = newText.replace(endStop, "")
            #Sets the tweet's text to the new value
            item['text'] = newText

    #removes any repeated tweets, and any retweets of existing tweets
    def removeRepeats(self):
        repeatArray = np.array([])
        #Create a copy of the dataframe
        myCopy = copy.copy(self.cleanedJson['data'])
        #Iterates through each tweet. 
        for item in myCopy:
            #If the tweet includes 'rt @' (indicates retweet) or if the tweet's text has already been added to the dataframe, removes it.
            if ('rt @' in item['text']):
                self.cleanedJson["data"].remove(item)
            elif item['text'] in repeatArray:
                self.cleanedJson["data"].remove(item)
            else:
                #If the tweet is new, adds it to the repeatArray to be checked against future tweets. 
                repeatArray = np.append(repeatArray, item["text"])


    #removes stopwords, according to nltk's list of stopwords
    def removeStopWords(self):
        #Get list of stopwords
        nltk.download('stopwords')
        stopWords = stopwords.words('english')
        #Iterate through each tweet
        for item in self.cleanedJson["data"]:
            tweetText = item['text']
            textArr = tweetText.split(' ')
            copyText = copy.copy(textArr)
            #iterates through each word, and checks if it is a stop word. IF it is, removes it before rejoining the words into one string.
            for word in copyText:
                if word in stopWords:
                    textArr.remove(word)
            newText = ' '.join(textArr)
            #Updates tweet text to exclude stop words
            item['text'] = newText

    #Utility function. Calls all 4 maintenance cleaning functions in tweetCleaner. 
    #Can be used to immediately prepare data for modelAnalyzer
    def prepTweets(self):
        self.lowerTweets()
        self.removeLinks()
        self.removeRepeats()
        self.removeStopWords()


