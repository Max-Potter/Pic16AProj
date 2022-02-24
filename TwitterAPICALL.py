import requests
import os
import json
import APISecret

#print("hello")

#print(os.environ.get("API_KEY"))


def urlGen(search, maxTweets = 10):
    search = search + " lang:en"
    url = "https://api.twitter.com/2/tweets/search/recent"

    query_params = {'query': search,
                    'max_results': maxTweets,
                    'next_token': {}}
    return url, query_params


def v2auth():
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

def callTwitter():
    bearer_token = v2auth()
    headers = requestHeaders(bearer_token)
    search = "Kanye Pete"
    max_results = 20
    url = urlGen(search, max_results)
    json_response = accessEndpoint(url[0], headers, url[1])
    return json_response


jsonresponse = callTwitter()
for item in jsonresponse["data"]:
    print(item["text"])
print("done")