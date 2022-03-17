import TwitterAPICALL
import tweetCleaner
from tweetCleaner import tweetCleaner
import matplotlib
from matplotlib import pyplot as plt
import json
import modelAnalyzer
from modelAnalyzer import modelAnalyzer
import sklearn
from sklearn import tree

def show_Tree(T,X, name):
    fig, ax = plt.subplots(1, figsize = (10, 10))
    p = tree.plot_tree(T, filled = True, feature_names = X.columns)
    ax.set(title = "Decision Tree for " + name)
    plt.show()

####
#myjson = TwitterAPICALL.getPastSevenDays("Bob Ross", 100)
#g = tweetCleaner(myjson)
#g.prepTweets()

#modelAnalysis = modelAnalyzer([])
#modelAnalysis.addJson(g.cleanedJson, "Bob_RossLikes")

#myjson = TwitterAPICALL.getPastSevenDays("Kanye Pete Davidson", 100)

#g = tweetCleaner(myjson)
#g.prepTweets()
#modelAnalysis.addJson(g.cleanedJson, "Kanye_PeteLikes")

#myjson = TwitterAPICALL.getPastSevenDays("Ukraine Russia", 100)

#g = tweetCleaner(myjson)
#g.prepTweets()
#modelAnalysis.addJson(g.cleanedJson, "Ukraine_RussiaLikes")
####

modelAnalysis = modelAnalyzer(['Bob_Ross.json','Kanye_Pete.json','Ukraine_Russia.json'])
fig = modelAnalysis.fitAllData(minLikes = 10)
plt.show()

allData = modelAnalysis.getAllData(minLikes = 10)
(X,y) = allData['Bob_Ross.json']

best_depth, fig = modelAnalysis.best_fit_Tree(X, y, max_tree_depth = 12)
fig.suptitle("Bob Ross  Dataset")
print(best_depth)
plt.show()
T, X_train, X_test, y_train, y_test = modelAnalysis.fit_tree(X, y, max_depth = best_depth)

show_Tree(T, X, "Bob Ross Data")

