import TwitterAPICALL




# numTweetsWithWord = {"best" : 5, "worst" : 7}
# wordMatrixx -->    

#   (CurrentAvg * numTweetsWithThatWord (doesn't include currentTweet) )) + CurrentTweetsLikeCount
#    ----------------------------------------------------------
#               numTweetsWithThatWord + 1



# same matrix, we take every word it has a '1' or a '0' for each tweet as a row
# in each row of that same dataframe, we want an extra column that is the like count