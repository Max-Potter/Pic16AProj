import requests
import os
import json
import APISecret #Remove this line to test

#print("hello")

#print(os.environ.get("API_KEY"))


def urlGen(search, maxTweets = 10):
    search = search + " lang:en"
    url = "https://api.twitter.com/2/tweets/search/recent"

    query_params = {'query': search,
                    'max_results': maxTweets,
                    'next_token': {},
                    'tweet.fields': 'organic_metrics'}
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

def callTwitter(search, max_results=20):
    bearer_token = v2auth()
    headers = requestHeaders(bearer_token)
    #search = "Kanye Pete"
    #max_results = 20
    url = urlGen(search, max_results)
    json_response = accessEndpoint(url[0], headers, url[1])
    return json_response


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
