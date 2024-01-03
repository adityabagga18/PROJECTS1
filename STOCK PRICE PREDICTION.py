#!/usr/bin/env python
# coding: utf-8

# In[28]:


import yfinance as yf
import pandas as pd


# In[2]:


sp500=yf.Ticker("^GSPC")


# In[3]:


sp500=sp500.history(period="max") # to fetch the historical data


# In[4]:


sp500


# In[5]:


sp500.index


# In[6]:


sp500.plot.line(y="Close", use_index=True)


# In[7]:


del sp500["Dividends"]
del sp500["Stock Splits"]


# In[8]:


sp500


# In[10]:


sp500["Tomorrow"]=sp500["Close"].shift(-1)


# In[11]:


sp500


# In[14]:


sp500["Target"]=(sp500["Tomorrow"]> sp500["Close"]).astype(int)


# In[15]:


sp500


# In[20]:


sp500=sp500.loc["1990-01-01":].copy()


# In[21]:


sp500


# In[23]:


from sklearn.ensemble import RandomForestClassifier

model= RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train=sp500[:-100] # all exluding ending 100 obs
test= sp500.iloc[-100:] # last 100 obs


predictors=['Close','Volume', 'High', 'Low']
model.fit(train[predictors], train["Target"])


# In[39]:


from sklearn.metrics import precision_score

pred=model.predict(test[predictors])


# In[40]:


pred=pd.Series(pred,index=test.index)


# In[41]:


pred


# In[43]:


precision_score(test["Target"], pred)


# In[65]:


horizons = [2, 5, 60, 250, 1000]  # Trading days
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500['Close'].rolling(horizon).mean()

    ratio_column = f"_close_ratio-{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages  # Corrected calculation

    trend_column = f"trend_{horizon}"
    
    sp500[trend_column] = sp500['Target'].shift(1).rolling(horizon).sum()  # Corrected calculation

    new_predictors += [ratio_column, trend_column]


# In[68]:


sp500


# In[69]:


sp500=sp500.dropna()


# In[70]:


sp500


# In[71]:


model=RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


# In[82]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    pred_proba = model.predict_proba(test[predictors])
    
    if pred_proba.shape[1] > 1:
        pred = pred_proba[:, 1]
    else:
        pred = pred_proba.flatten()
    
    pred[pred >= 0.6] = 1
    pred[pred < 0.6] = 0
    
    pred = pd.Series(pred, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], pred], axis=1)
    return combined



# In[ ]:





# In[78]:


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:1]
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)


# In[83]:


predictions=backtest(sp500,model,new_predictors)


# In[84]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[ ]:





# In[ ]:




