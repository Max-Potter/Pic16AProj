import TwitterAPICALL
import nltk
from nltk.corpus import stopwords
import numpy as np
import copy


class tweetCleaner():
    def __init__(self, jsonObj):
        self.json = jsonObj
        self.cleanedJson = copy.copy(jsonObj)

    def lowerTweets(self):
        for item in self.cleanedJson['data']:
            item['text'] = item['text'].lower()

    def removeLinks(self):
        endStop = r"\n"
        for item in self.cleanedJson['data']:
            tweetText = item['text']
            textArr = tweetText.split(' ')
            textCopy = copy.copy(textArr)
            for word in textCopy:
                if 'http' in word:
                    textArr.remove(word)
            newText = ' '.join(textArr)
            newText = newText.replace(endStop, "")
            item['text'] = newText

    def removeRepeats(self):
        repeatArray = np.array([])
        myCopy = copy.copy(self.cleanedJson['data'])
        for item in myCopy:
            if ('rt @' in item['text']):
                self.cleanedJson["data"].remove(item)
            elif item['text'] in repeatArray:
                self.cleanedJson["data"].remove(item)
            else:
                repeatArray = np.append(repeatArray, item["text"])

    def removeStopWords(self):
        nltk.download('stopwords')
        stopWords = stopwords.words('english')
        for item in self.cleanedJson["data"]:
            tweetText = item['text']
            textArr = tweetText.split(' ')
            copyText = copy.copy(textArr)
            for word in copyText:
                if word in stopWords:
                    textArr.remove(word)
            newText = ' '.join(textArr)
            item['text'] = newText

    def prepTweets(self):
        self.lowerTweets()
        self.removeLinks()
        self.removeRepeats()
        self.removeStopWords()



#json = TwitterAPICALL.callTwitter("Kanye Pete Davidson beef", 100)
#json = TwitterAPICALL.getPastSevenDays("Kanye Pete Davidson", 100)
#print(json)
#print(type(json))
#g = tweetCleaner(json)
#for item in g.cleanedJson['data']:
  #  print(item['text'],"DONE")
#g.prepTweets()
#for item in g.cleanedJson['data']:
##    print(item['public_metrics'],"DONE")
#print(len(g.cleanedJson['data']))


#print("done")