import yfinance as yf
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import numpy as np

from functions import backtest, importData

stock = "MSFT"
stockHistory = importData(stock)
stockHistory.plot.line(y="Close", use_index=True)

#ensure we use the actual close price
data = stockHistory[["Close"]]
data = data.rename(columns={"Close": "ActualClose"})

#target identifies if the price went up or down
data["Target"] = stockHistory.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

#need to shift prices forward to avoid look ahead bias
stockPrev = stockHistory.copy()
stockPrev = stockHistory.shift(1)

predictors = ["Close", "Volume", "Open", "High", "Low"]
data = data.join(stockPrev[predictors]).iloc[1:]

#calculate rolling means
weeklyMean = data.rolling(7).mean()["ActualClose"]
quarterlyMean = data.rolling(90).mean()["ActualClose"]
annualMean = data.rolling(365).mean()["ActualClose"]

weeklyTrend = data.shift(1).rolling(7).sum()["Target"]

data["weeklyMean"] = weeklyMean / data["ActualClose"]
data["quarterlyMean"] = quarterlyMean / data["ActualClose"]
data["annualMean"] = annualMean / data["ActualClose"]

data["annualWeeklyMean"] = data["annualMean"] / data["weeklyMean"]
data["annualQuarterlyMean"] = data["annualMean"] / data["quarterlyMean"]

data["weeklyTrend"] = weeklyTrend

data["openCloseRatio"] = data["Open"] / data["ActualClose"]
data["highCloseRatio"] = data["High"] / data["ActualClose"]
data["lowCloseRatio"] = data["Low"] / data["ActualClose"]

#creating the training data
fullPredictors = predictors + ["weeklyMean", "quarterlyMean", "annualMean", "annualWeeklyMean", "annualQuarterlyMean", "openCloseRatio", "highCloseRatio", "lowCloseRatio"]


#initialize a radom forest classifier. High min samples split to avoid overfitting
model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)

predictions = backtest(data.iloc[365:], model, fullPredictors)


print(predictions["Predictions"].value_counts())
print(predictions["Target"].value_counts())
print(precision_score(predictions["Target"], predictions["Predictions"]))

predictions.iloc[-100:].plot()
plt.show()

