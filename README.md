# Pic16AProj
Project for Pic16A, Winter 2022 UCLA

Project Name: Twitter Sentiment Analyzer
Group Members: Tyler Nguyen, Max Potter, Brandon Achugbue

Short Description:
Using the NLTK Sentiment Analyzer and the Twitter API, this project will collect and analyze
user sentiments about various topics from their twitter posts. From these tweets, we will predict which tweets are more popular. 
Data will be further processed and analyzed using Scikit-learn to construct models and visualizations of these sentiments, and we specifically used 
the Decision Tree Classifier and lasso models. 

Instructions For Package Installation:
...

Detailed Description of Demo File:
...

Scope and Limitations: There were a few limitations to our project. Firstly, the main limitation was the infinitely large amounts of confounding variables. Popularity of tweets is based on many factors, such as follower count. We could not take all these variables into account, therefore our model did very poorly. In addition, another limitation was the NLTK sentiment analyzer is innately biased. This is because this model is a form of supervised learning, as the words are first classified into positive and negative sentiments by a human. For instance, some words may be  considered negative while other words may be considered positive. Additionally, because we used a term-document matrix in order to classify Tweets as either positive or negative, irony and sarcasm was hard to pick up on. For these reasons, the sentiment score may not accurately reflect what the author of the tweet truly meant as the tone was not picked up on. 
...

License and Terms of Use:
MIT License

References and Acknowledgement: 
- COMM 155 - Artificial Intelligence and New Media, Winter Quarter 2022, Prof. J. Joo, TA. Jacqueline Lam
  - Copy_of_COMM155_W22_Week_5_CSV_format_and_NLTK.ipynb

...

Background Source of Dataset:
...

Links to Tutorials:
- Online tutorials involved:
    - Text Analytics for Beginners using NLTK https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
    - Comparing Sentiment Analysis Tools https://investigate.ai/investigating-sentiment-analysis/comparing-sentiment-analysis-tools/#:~:text=What%27s%20this%20mean%20for%20me%3F%20%23%20%20,words%20%20%20automatic%20based%20on%20score%20
    - Sentiment Analysis Using NLTK https://medium.com/analytics-vidhya/sentiment-analysis-using-nltk-d520f043fc0

...


