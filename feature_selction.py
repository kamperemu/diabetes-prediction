
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
from helper import loadXY

def feature_selection():
    X, Y = loadXY()
    bestfeatures = SelectKBest(score_func=chi2, k=8)
    fit = bestfeatures.fit(X,Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']
    return featureScores.nlargest(8,'Score')

if __name__ == "__main__":
    print(feature_selection())