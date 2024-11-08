import yfinance as yf
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import numpy as np


def importData(ticker):
    DATA_PATH = "stockData.json"

    #read the stock data from a file if it exists, otherwise download it
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH) as f:
            stockHistory = pd.read_json(DATA_PATH)

    else:
        stock = yf.Ticker(ticker)
        stockHistory = stock.history(period="max")

        stockHistory.to_json(DATA_PATH)
    
    return stockHistory

def backtest(data, model, predictors, start=1000, step=350):
    predictions = []
    #loop through the dataset incrementally
    for i in range(start, data.shape[0],step):
        #split between training and testing data
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()

        #fit with random forest model
        model.fit(train[predictors], train["Target"])

        #make predictions
        preds = model.predict_proba(test[predictors])[:,1]
        preds = pd.Series(preds, index=test.index)
        preds[preds > .6] = 1
        preds[preds <= .6] = 0 

        #combine predictions and tests
        combined = pd.concat({"Target": test["Target"], "Predictions": preds}, axis=1)

        predictions.append(combined)

    print(predictions[0].head())

    return pd.concat(predictions)