import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('vader_lexicon')

sentences = ["list of sentences","sentence 2","oo angry sentence"]

sid = SentimentIntensityAnalyzer()

for sentence in sentences:
    score=sid.polarity_scores(sentence)
    print(sentence,score)