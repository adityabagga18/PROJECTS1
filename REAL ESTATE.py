#!/usr/bin/env python
# coding: utf-8

# # REAL ESTATE

# In[18]:


import pandas as pd


# In[24]:


fed_files = ["/Users/aditya/Downloads/house_prices/MORTGAGE30US.csv", "/Users/aditya/Downloads/house_prices/RRVRUSQ156N.csv", "/Users/aditya/Downloads/house_prices/CPIAUCSL.csv"]

dfs = [pd.read_csv(f, parse_dates=True, index_col=0) for f in fed_files]



# In[25]:


dfs[0]


# In[10]:


dfs[1]


# In[11]:


dfs[2]


# In[12]:


fed_data=pd.concat(dfs,axis=1)


# In[13]:


fed_data


# In[14]:


fed_data.tail(50)


# # we will use forward fill

# In[15]:


fed_data=fed_data.ffill()


# In[17]:


fed_data.tail(50)


# In[26]:


zillow_files=["/Users/aditya/Downloads/house_prices/Metro_median_sale_price_uc_sfrcondo_week.csv","/Users/aditya/Downloads/house_prices/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv"]


# In[27]:


dfs = [pd.read_csv(z, parse_dates=True, index_col=0) for z in zillow_files]


# In[28]:


dfs[0]


# In[30]:


dfs=[pd.DataFrame(df.iloc[0,5:])for df in dfs]


# In[31]:


dfs[0]


# In[32]:


dfs[1]


# In[40]:


for df in dfs:
    df.index=pd.to_datetime(df.index)
    df["month"]=df.index.to_period("M")


# In[41]:


dfs[0]


# In[43]:


dfs[1]


# In[44]:


price_data=dfs[0].merge(dfs[1],on="month")


# In[46]:


price_data.index=dfs[0].index


# In[47]:


price_data


# In[48]:


del price_data["month"]


# In[49]:


price_data.columns=["price","value"]


# In[50]:


price_data


# In[51]:


fed_data=fed_data.dropna()


# In[52]:


fed_data


# In[53]:


fed_data.tail(20)


# In[54]:


from datetime import timedelta

fed_data.index=fed_data.index + timedelta(days=2)


# In[57]:


fed_data


# In[58]:


price_data=fed_data.merge(price_data, left_index=True, right_index= True)


# In[59]:


price_data


# In[60]:


price_data.columns=["Interest","vacancy","cpi","price","value"]


# In[62]:


price_data


# In[63]:


price_data.plot.line(y="price", use_index=True)


# In[64]:


price_data["adj_price"]=price_data['price']/price_data["cpi"]*100


# In[65]:


price_data.plot.line(y="adj_price", use_index=True)


# In[66]:


price_data["adj_value"]=price_data["value"]/price_data["cpi"]*100


# In[67]:


price_data.plot.line(y="adj_value", use_index=True)


# In[68]:


price_data["next_quarter"]=price_data['adj_price'].shift(-13) #For instance, in finance, 
                                                            #quarterly reports are released every three months, which translates to approximately 13 weeks (considering 4 quarters in a year).


# In[69]:


price_data


# In[70]:


price_data.dropna(inplace=True)


# In[71]:


price_data


# In[73]:


price_data["change"]=(price_data["next_quarter"]>price_data["adj_price"]).astype(int)


# In[74]:


price_data


# In[78]:


price_data["change"].value_counts()


# In[88]:


predictors=["Interest","vacancy","adj_price","adj_value"]

target= "change"


# In[81]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# In[89]:


def predict(train,test,predictors,target):
    rf=RandomForestClassifier(min_samples_split=10, random_state=1)
    rf.fit(train[predictors],train[target])
    pred=rf.predict(test[predictors])
    return pred


start=260 # 5 year
step=52 # 52 weeks in year
def backtest(data,predictors,target):
    all_pred=[]
    for i in range(start , data.shape[0], step):
        train=price_data.iloc[:i]
        test=price_data.iloc[i:(i+step)]
        all_pred.append(predict(train,test,predictors,target))
        
    pred=np.concatenate(all_pred)   
    return pred, accuracy_score(data.iloc[start:][target], pred)


# - protects against overfitting min split

# In[90]:


pred,accuracy=backtest(price_data,predictors,target)


# In[91]:


accuracy


# In[92]:


yearly= price_data.rolling(52, min_periods=1).mean()


# In[93]:


yearly


# In[97]:


yearly_ratios=[p + '_year' for p in predictors]
price_data[yearly_ratios]=price_data[predictors]/yearly[predictors]


# In[95]:


price_data


# In[100]:


pred,accuracy=backtest(price_data,predictors+yearly_ratios,target)


# In[101]:


accuracy


# In[111]:


from sklearn.inspection import permutation_importance

rf= RandomForestClassifier(min_samples_split=10, random_state=1)
rf.fit(price_data[predictors], price_data[target])

results=permutation_importance(rf,price_data[predictors], price_data[target], n_repeats=10, random_state=1)


# In[114]:


results["importances_mean"]


# In[115]:


predictors


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




