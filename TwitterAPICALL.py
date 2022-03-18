import requests
import os
import json
import APISecret
import datetime
from datetime import datetime, timedelta


def urlGen(search, maxTweets = 10, hours_before = 0):
    search = search + " lang:en" + " -is:retweet"
    url = "https://api.twitter.com/2/tweets/search/recent"
    endTime = datetime.utcnow()
    endTime = endTime - timedelta(hours = hours_before)
    endTime = endTime - timedelta(seconds=30)
    endTime = str(endTime)
    endTime = endTime.replace(' ', 'T')
    endTime = endTime + 'Z'

    query_params = {'query': search,
                    'end_time': endTime,
                    'max_results': maxTweets,
                    'next_token': {},
                    'tweet.fields': 'public_metrics'}
    return url, query_params


def v2auth():
    return os.environ.get("BEARER_TOKEN")

def requestHeaders(token):
    headers = {"Authorization": "Bearer {}".format(token)}
    return headers

def accessEndpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token
    response = requests.request("GET", url, headers=headers, params=params)
    #print("Response: " + str(response.status_code))
    if str(response.status_code) == "429":
        raise Exception("Error -- Twitter Rate Limit reached. Rate limit refreshes every 15 minutes, try again later.")

    return response.json()

def callTwitter(search, max_results=20, hours_before = 0):
    bearer_token = v2auth()
    headers = requestHeaders(bearer_token)
    url = urlGen(search, max_results, hours_before)
    json_response = accessEndpoint(url[0], headers, url[1])
    return json_response

def getPastSevenDays(search, max_results = 20):
    print("Beginning Tweet Retrieval - This may take up to a few minutes depending on volume")
    initJson = callTwitter(search, max_results, 0)
    for hourVal in range(0, 168, 2):
        newJson = callTwitter(search, max_results, hourVal)
        try:
            for item in newJson['data']:
                initJson['data'].append(item)
        except KeyError:
            raise KeyError("Error: Twitter API returned a bad request. You may have hit the rate limit, or sent a search query that returned no results.")
        except:
            raise Exception("Error: Something went wrong and the API request returned invalid data. Try a different search query.")
        #finalJson = updateJson
        #finalJson.update(callTwitter(search, max_results=max_results, hours_before = hourVal))
    print("Retrieved all tweets with no issues")
    return initJson

