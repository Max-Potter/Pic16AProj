import requests
import os
import json
import APISecret
import datetime
from datetime import datetime, timedelta


#Constructs the url to make the api request with.
#@params
#search: A string. The intended search query for tweets
#@kwargs
#maxTweets: An integer. The maximum number of tweets to retrieve. Must be between 10-100
#hours_before: An integer. The minimum number of hours since the tweet has been made to be included.
    #This value must be between 0 and 168
#Returns two values: url and query_params
def urlGen(search, maxTweets = 10, hours_before = 0):
    if maxTweets > 100 or maxTweets < 10:
        raise Exception("Error: You must select a maxTweets value between 10 and 100")
    if hours_before > 168:
        raise Exception("Error: You are attempting to call tweets too far in the past. hours_before must be less than 169.")
    elif hours_before < 0:
        raise Exception("Error: hours_before must be positive")

    #Specifies to only retrieve english tweets and exclude retweets
    search = search + " lang:en" + " -is:retweet"
    #Base url for twitter API recent search requests
    url = "https://api.twitter.com/2/tweets/search/recent"

    #Converts hours_before into a date that can be passed to twitter API.
    endTime = datetime.utcnow()
    endTime = endTime - timedelta(hours = hours_before)
    endTime = endTime - timedelta(seconds=30)
    endTime = str(endTime)
    endTime = endTime.replace(' ', 'T')
    endTime = endTime + 'Z'

    #Adds the parameters of the query.
    #Query -- the search
    #end_time -- hours_before
    #max_results -- max number of tweets
    #tweet.fields -- additional tweet info to return

    query_params = {'query': search,
                    'end_time': endTime,
                    'max_results': maxTweets,
                    'next_token': {},
                    'tweet.fields': 'public_metrics'}
    return url, query_params

#Retrieves authorization token used in API call
def v2auth():
    return os.environ.get("BEARER_TOKEN")

#Formats headers for the api request, using the authorization token to gain access.
def requestHeaders(token):
    headers = {"Authorization": "Bearer {}".format(token)}
    return headers

#Accesses the twitter API and retrieves a json object containing tweets matching specified inputs
#@params
#url -- url of the api request. Provided by urlGen() function
#headers -- headers for the api request. Provided by requestHeaders() function
#params -- Parameters of the query. Returned with the urlGen() function
#next_token -- used to iterate through tweets on API
#Returns a json object of the twitter api response
def accessEndpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token
    #Make the API request
    response = requests.request("GET", url, headers=headers, params=params)
    #print("Response: " + str(response.status_code))
    #Checks that the API did not return a rate Limit exceeded error
    if str(response.status_code) == "429":
        raise Exception("Error -- Twitter Rate Limit reached. Rate limit refreshes every 15 minutes, try again later.")

    return response.json()

#Convenient function to automatically retrieve the twitter json given a search query
#Calls the other functions in this file automatically
#@params
#search: A string. The search query.
#@kwargs
#max_results: an integer. Maximum number of results. Default 20, but must be between 10 and 100. 
#hours_before: An integer. The minimum number of hours since the tweet has been made to be included.
    #This value must be between 0 and 168
def callTwitter(search, max_results=20, hours_before = 0):
    #get authorization token
    bearer_token = v2auth()
    #construct headers with auth token
    headers = requestHeaders(bearer_token)
    #generate the api call url using search query and additional parameters
    url = urlGen(search, max_results, hours_before)
    #Make the API call
    json_response = accessEndpoint(url[0], headers, url[1])
    return json_response


#Convenient function to automatically return a large number of results.
#Gets the 100 most recent tweets from every 2 hour interval in the past 7 days.
#Concatenates all json objects returned into a single json object. 
#@params
#search: A string. The search query.
#@kwargs
#max_results: An integer. Maxmimum number of results. Default 20, but must be between 10 and 100.
#Returns the complete json of all responses.
def getPastSevenDays(search, max_results = 20):
    print("Beginning Tweet Retrieval - This may take up to a few minutes depending on volume")
    #Initialize Json to the first api call
    initJson = callTwitter(search, max_results, 0)
    #Iterates through every 2 hour interval in past 7 days
    for hourVal in range(0, 168, 2):
        #Makes a new API call
        newJson = callTwitter(search, max_results, hourVal)
        #Updates the current initJson to include the new json's data.
        try:
            for item in newJson['data']:
                initJson['data'].append(item)
        except KeyError: #Triggers if twitter does not return a valid json object
            raise KeyError("Error: Twitter API returned a bad request. You may have hit the rate limit, or sent a search query that returned no results.")
        except: #Triggers if another error occurs
            raise Exception("Error: Something went wrong and the API request returned invalid data. Try a different search query.")
        #finalJson = updateJson
        #finalJson.update(callTwitter(search, max_results=max_results, hours_before = hourVal))
    print("Retrieved all tweets with no issues")
    return initJson

