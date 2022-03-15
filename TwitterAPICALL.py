import requests
import os
import json
import APISecret #Remove this line to test
import datetime
from datetime import datetime, timedelta

#print("hello")

#print(os.environ.get("API_KEY"))


def urlGen(search, maxTweets = 10, hours_before = 0):
    search = search + " lang:en" + " -is:retweet"
    url = "https://api.twitter.com/2/tweets/search/recent"
    endTime = datetime.utcnow()
    endTime = endTime - timedelta(hours = hours_before)
    endTime = endTime - timedelta(seconds=30)
    endTime = str(endTime)
    endTime = endTime.replace(' ', 'T')
    endTime = endTime + 'Z'
    #print(endTime)

    query_params = {'query': search,
                    'end_time': endTime,
                    'max_results': maxTweets,
                    'next_token': {},
                    'tweet.fields': 'public_metrics'}
    return url, query_params


def v2auth():
    #REPLACE THIS LINE WITH BEARER TOKEN TO TEST
    #ALSO REMOVE THE LINE "import APISecret"
    return os.environ.get("BEARER_TOKEN")

def requestHeaders(token):
    headers = {"Authorization": "Bearer {}".format(token)}
    return headers

def accessEndpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token
    response = requests.request("GET", url, headers=headers, params=params)
    #print(os.environ.get("BEARER_TOKEN"))
    print("Response: " + str(response.status_code))

    return response.json()

def callTwitter(search, max_results=20, hours_before = 0):
    bearer_token = v2auth()
    headers = requestHeaders(bearer_token)
    #search = "Kanye Pete"
    #max_results = 20
    url = urlGen(search, max_results, hours_before)
    json_response = accessEndpoint(url[0], headers, url[1])
    return json_response

def getPastSevenDays(search, max_results = 20):
    initJson = callTwitter(search, max_results, 0)
    for hourVal in range(0, 168, 2):
        newJson = callTwitter(search, max_results, hourVal)
        for item in newJson['data']:
            initJson['data'].append(item)
        #finalJson = updateJson
        #finalJson.update(callTwitter(search, max_results=max_results, hours_before = hourVal))
    return initJson



#jsonresponse = callTwitter("Kanye Pete Davidson beef", 40)
#HistDict = {}
#for item in jsonresponse["data"]:
#    if 'RT @' not in item["text"]:
#        try:
#            HistDict[item["text"][0:12]]
#        except:
#            HistDict[item["text"][0:12]] = 1
#            print(item["text"])
        
#print("done")
#print(jsonresponse)
#print(HistDict["hey"])
