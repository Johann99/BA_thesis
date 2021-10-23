#!/usr/bin/env python
# coding: utf-8

# In[1]:


#  === Workspace preparation ===

# import packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import Counter

import scipy 
from scipy import stats


import csv
import os
import datetime as dt
import yfinance as yf

# NLTK VADER for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

# granger causality analysis 
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller

# Vector Autoregression Model
from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import VARMAX
from statsmodels.tsa.api import AutoReg


# In[2]:


get_ipython().run_line_magic('who', '')


# In[3]:


import types
def imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            yield val.__name__
list(imports())


# # Business Understanding
# In the following analysis I am going to exmaine whether there is a relation betweet the public sentiment expressed on twitter regarding certain stocks and the actual stock price. 
# The following stocks are part of this analysis: Apple, Amazon, Microsoft, Tesla 
# 
# ***

# # Data Understanding
# 
# This dataset was first published in a paper at the 2020 IEEE International Conference on Big Data under the 6th Special Session on Intelligent Data Mining track. I checked also their scraping code and contacted one of the authors in order to verifiy it's credibility. 
# 
# *** 
# source: https://www.kaggle.com/omermetinn/tweets-about-the-top-companies-from-2015-to-2020
# 
# source: https://github.com/omer-metin/TweetCollector
# 
# source: https://www.yahoofinanceapi.com/gclid=CjwKCAjwx8iIBhBwEiwA2quaq7mzIHmf8CJmSk8z6KJnUUQrZd_oBF_OllZXKRwDT1BCrJGYHJpNVxoCg0AQAvD_BwE
# 
# source: https://ieeexplore.ieee.org/document/9378170

# # Data Import  

# In[4]:


#  === Tweets ===
tweets = pd.read_csv("/Users/pietj.ginski/Desktop/BWL-Studium/BWL 6 Semester/Bachelor Thesis/Raw Data BT/archive-2/Kaggle/Tweet.csv")

#
tweets['date_s'] = pd.to_datetime(tweets.post_date,unit='s').dt.strftime('%d-%m-%Y')

# convert to date 
tweets.date_s = pd.to_datetime(tweets.date_s)

# filter for more popular tweets 
# tweets = tweets[tweets['retweet_num']>3]


# In[5]:


# check data 
tweets.head()


# In[6]:


# how many indiviudal writers? and how many tweets?
print('I collected',tweets['tweet_id'].count(), 'tweets from', tweets['writer'].nunique(), 'unique writers.')


# In[7]:


# convert to date and create date columns  
tweets['date'] = pd.to_datetime(tweets.post_date,unit='s').dt.strftime('%d-%m-%Y %H:%M:%S')
tweets['weekday'] = pd.to_datetime(tweets.post_date,unit='s').dt.strftime('%a')
tweets['year'] = pd.to_datetime(tweets.post_date,unit='s').dt.strftime('%Y')


# In[8]:


#  === Company_Tweet ===
company_tweets = pd.read_csv("/Users/pietj.ginski/Desktop/BWL-Studium/BWL 6 Semester/Bachelor Thesis/Raw Data BT/archive-2/Kaggle/Company_Tweet.csv")

# check data 
company_tweets.head()


# In[9]:


# merge the two datasets
tweets_merged = pd.merge(company_tweets, tweets, on='tweet_id', how='inner')

# check data 
tweets_merged.tail()


# In[10]:


#  === Stock Data ===


# In[11]:


# *** Stock Data Function ***

def yahoo(ticker, start_date, end_date, df_name):

# define the ticker symbol
    tickerSymbol = ticker

# get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)

# get the historical prices for this ticker
    df_name = tickerData.history(period='1d', start=start_date, end=end_date)

# get daily returns
    df_name["daily_returns"] = df_name["Close"].pct_change()

# convert index to date_s colums
    df_name['date_s'] = pd.to_datetime(df_name.index)
    
    return df_name


# In[12]:


# apple stock data
apple_p = yahoo('AAPL','2015-12-31','2018-12-31','apple_p')

# amazon stock data 
amazon_p = yahoo('AMZN','2015-12-31','2018-12-31','amazon_p')

# microsoft stock data 
microsoft_p = yahoo('MSFT','2015-12-31','2018-12-31','microsoft_p')

# tesla stock data 
tesla_p = yahoo('TSLA','2015-12-31','2018-12-31','tesla_p')


# In[13]:


print(apple_p.head(), amazon_p.head(), microsoft_p.head(), tesla_p.head())


# # Data Inspection and Visualisation

# ## Distribution of ø Retweets, Comment, Likes 

# In[14]:


# Distribution of Retweets 
r = pd.DataFrame(tweets['retweet_num'].groupby(tweets['writer']).agg('mean').sort_values())

influencer = r[r['retweet_num']>=4]

not_influencer = r[r['retweet_num']<4]


# In[15]:


plt.pie((len(influencer),len(not_influencer)),autopct='%.2f')
plt.legend(['>=4','< 4'])
plt.title('Anzahl an Retweets')

plt.savefig("retweet.png", dpi=460)


# In[16]:


# Distribution of Comments 
c = pd.DataFrame(tweets['comment_num'].groupby(tweets['writer']).agg('mean').sort_values())

influencer = c[c['comment_num']>=4]

not_influencer = c[c['comment_num']<4]


# In[17]:


plt.pie((len(influencer),len(not_influencer)),autopct='%.2f')
plt.legend(['>=4','< 4'])
plt.title('Anzahl an Comments')

plt.savefig("comment.png", dpi=460)


# In[18]:


# Distribution of Like 
l = pd.DataFrame(tweets['like_num'].groupby(tweets['writer']).agg('mean').sort_values())

influencer = l[l['like_num']>=4]

not_influencer = l[l['like_num']<4]


# In[19]:


plt.pie((len(influencer),len(not_influencer)),autopct='%.2f')
plt.legend(['>=4','< 4'])
plt.title('Anzahl an Comments')

plt.savefig("comment.png", dpi=460)


# ## Most frequent Writers 

# In[20]:


# top 40 most frequent writers
a = tweets['tweet_id'].groupby(tweets['writer']).nunique().sort_values(ascending=False).index[:40]
b = tweets['tweet_id'].groupby(tweets['writer']).nunique().sort_values(ascending=False)[:40]

plt.figure(figsize = (20, 8))
plt.xticks(rotation=45)
plt.bar(a,b)
plt.show()


# In[21]:


count = 0 
mean = tweets['tweet_id'].groupby(tweets['writer']).nunique().mean()

for i in tweets['tweet_id'].groupby(tweets['writer']).nunique():
    if i>mean:
        count = count + 1

print(round(count/tweets['writer'].nunique(), 3))


# ## Weekly Trends 

# In[22]:


# evolution of unique tweets per weekday 
x = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']


fig, ax1 = plt.subplots()

ax1.plot(x, tweets['comment_num'].groupby(tweets['date_s'].dt.weekday).agg('sum'))
ax1.plot(x, tweets['retweet_num'].groupby(tweets['date_s'].dt.weekday).agg('sum'))
ax1.plot(x, tweets['like_num'].groupby(tweets['date_s'].dt.weekday).agg('sum'))
ax1.legend(['comment_num', 'retweet_num', 'like_num'], loc = 'upper right',bbox_to_anchor=(1.0, 1))
plt.xticks(rotation=20)

ax2 = ax1.twinx()
ax2.plot(x, tweets['tweet_id'].groupby(tweets['date_s'].dt.weekday).agg('count'), color = 'grey')
ax2.legend(['Count'], loc = 'upper left')

plt.show()


# ## Total Number of Unique Tweets 

# In[23]:


# check amount of unique tweets
ap = tweets_merged[tweets_merged['ticker_symbol'] == 'AAPL']['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')
ts = tweets_merged[tweets_merged['ticker_symbol'] == 'TSLA']['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')
ms = tweets_merged[tweets_merged['ticker_symbol'] == 'MSFT']['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')
az = tweets_merged[tweets_merged['ticker_symbol'] == 'AMZN']['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')

year = ['2015','2016','2017','2018','2019']

plt.plot(year,ap,
        color = 'royalblue', marker='o')
plt.plot(year,ts,
        color = 'red', marker='o')
plt.plot(year,ms,
        color = 'darkblue', marker='o')
plt.plot(year,az,
        color = 'orange', marker='o')
plt.legend(['apple', 'tesla','microsoft','amazon'], loc='upper right')
plt.title('Total Number of Unique Tweets')
plt.show()


# ## Average Comment, Retweet, Like per Unique Tweet

# In[24]:


# check ø comment_num, retweet_num and like_num

comment_share_year = tweets_merged['comment_num'].groupby(tweets_merged['date_s'].dt.year).agg('sum')/tweets_merged['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')
retweet_share_year = tweets_merged['retweet_num'].groupby(tweets_merged['date_s'].dt.year).agg('sum')/tweets_merged['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')
likes_share_year = tweets_merged['like_num'].groupby(tweets_merged['date_s'].dt.year).agg('sum')/ tweets_merged['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')


plt.plot(year,comment_share_year, marker='o')
plt.plot(year,retweet_share_year, marker='o')
plt.plot(year,likes_share_year, marker='o')
plt.legend(['comment_num','retweet_num','like_num'])
plt.title('ø Comments, Retweets or Likes per Tweet')
plt.show()


# ## Average Comment per Unique Tweet

# In[25]:


# ø Comment per Unique Tweet
ap = tweets_merged[tweets_merged['ticker_symbol'] == 'AAPL']['comment_num'].groupby(tweets_merged['date_s'].dt.year).agg('sum')/tweets_merged['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')
ts = tweets_merged[tweets_merged['ticker_symbol'] == 'TSLA']['comment_num'].groupby(tweets_merged['date_s'].dt.year).agg('sum')/tweets_merged['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')
ms = tweets_merged[tweets_merged['ticker_symbol'] == 'MSFT']['comment_num'].groupby(tweets_merged['date_s'].dt.year).agg('sum')/tweets_merged['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')
az = tweets_merged[tweets_merged['ticker_symbol'] == 'AMZN']['comment_num'].groupby(tweets_merged['date_s'].dt.year).agg('sum')/tweets_merged['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')

# month = ['Jan','Feb','Mar','Apr','Mai','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

plt.plot(year,ap,
        color = 'royalblue', marker='o')
plt.plot(year,ts,
        color = 'red', marker='o')
plt.plot(year,ms,
        color = 'darkblue', marker='o')
plt.plot(year,az,
        color = 'orange', marker='o')
plt.legend(['apple', 'tesla','microsoft','amazon'], loc='upper left')
plt.title('ø Comment per Unique Tweet')
plt.show()


# > __Tesla became a very hot topic over the years.__

# ## Average Retweet per Unique Tweet

# In[26]:


# check amount of unique tweets
ap = tweets_merged[tweets_merged['ticker_symbol'] == 'AAPL']['retweet_num'].groupby(tweets_merged['date_s'].dt.year).agg('sum')/tweets_merged['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')
ts = tweets_merged[tweets_merged['ticker_symbol'] == 'TSLA']['retweet_num'].groupby(tweets_merged['date_s'].dt.year).agg('sum')/tweets_merged['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')
ms = tweets_merged[tweets_merged['ticker_symbol'] == 'MSFT']['retweet_num'].groupby(tweets_merged['date_s'].dt.year).agg('sum')/tweets_merged['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')
az = tweets_merged[tweets_merged['ticker_symbol'] == 'AMZN']['retweet_num'].groupby(tweets_merged['date_s'].dt.year).agg('sum')/tweets_merged['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')

# month = ['Jan','Feb','Mar','Apr','Mai','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

plt.plot(year,ap,
        color = 'royalblue', marker='o')
plt.plot(year,ts,
        color = 'red', marker='o')
plt.plot(year,ms,
        color = 'darkblue', marker='o')
plt.plot(year,az,
        color = 'orange', marker='o')
plt.legend(['apple', 'tesla','microsoft','amazon'], loc='upper right')
plt.title('ø Retweet per Unique Tweet')
plt.show()


# ## Average Like per Unique Tweet

# In[27]:


# check amount of unique tweets
ap = tweets_merged[tweets_merged['ticker_symbol'] == 'AAPL']['like_num'].groupby(tweets_merged['date_s'].dt.year).agg('sum')/tweets_merged['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')
ts = tweets_merged[tweets_merged['ticker_symbol'] == 'TSLA']['like_num'].groupby(tweets_merged['date_s'].dt.year).agg('sum')/tweets_merged['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')
ms = tweets_merged[tweets_merged['ticker_symbol'] == 'MSFT']['like_num'].groupby(tweets_merged['date_s'].dt.year).agg('sum')/tweets_merged['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')
az = tweets_merged[tweets_merged['ticker_symbol'] == 'AMZN']['like_num'].groupby(tweets_merged['date_s'].dt.year).agg('sum')/tweets_merged['tweet_id'].groupby(tweets_merged['date_s'].dt.year).agg('count')

# month = ['Jan','Feb','Mar','Apr','Mai','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

plt.plot(year,ap,
        color = 'royalblue', marker='o')
plt.plot(year,ts,
        color = 'red', marker='o')
plt.plot(year,ms,
        color = 'darkblue', marker='o')
plt.plot(year,az,
        color = 'orange', marker='o')
plt.legend(['apple', 'tesla','microsoft','amazon'], loc='upper right')
plt.title('ø Like per Unique Tweet')
plt.show()


# In[28]:


# delete dataset 
del company_tweets
del tweets


# In[29]:


for f, b in zip(Counter(tweets_merged['ticker_symbol']).keys(), Counter(tweets_merged['ticker_symbol']).values()):
    print(f,'Proportion of Dataset:', round(b/len(tweets_merged), 2), '%')


# In[30]:


# display distribution in pie chart 
list_x = []
for i in Counter(tweets_merged['ticker_symbol']).values():
             list_x.append(i/len(tweets_merged))

# plot 
plt.pie(list_x, labels = Counter(tweets_merged['ticker_symbol']).keys(), autopct='%1.2f',)
plt.title('Propotion of Dataset in %')
plt.show() 


# In[31]:


# Preprocessing

# # remove $ 
# tweets_merged['new_body'] = tweets_merged['body'].str.replace('($\w+.*?)',"")

# # remove $ 
# tweets_merged['new_body'] = tweets_merged['new_body'].str.replace('(\$\w+.*?)',"")

# # remove @ 
# tweets_merged['new_body'] = tweets_merged['new_body'].str.replace('(@\w+.*?)',"")

# # remove @ 
# tweets_merged['new_body'] = tweets_merged['new_body'].str.replace('(\@\w+.*?)',"")

# # remove urls  
# tweets_merged['new_body'] = tweets_merged['new_body'].str.replace('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)',"")

# # check data 
# tweets_merged['new_body'].tail()


# In[34]:


# check data - index 20 has hashtags, links and mentions in one text
# tweets_merged['new_body'][20]


# # Data Modeling 

# In[38]:


#  === VADER ===

# create the analyzer 
analyzer = SentimentIntensityAnalyzer()

# prepare the data
sentences = np.array(tweets_merged["body"])

# get the vader scores 
tweets_merged['vader_scores'] = tweets_merged['body'].apply(lambda sentences: analyzer.polarity_scores(sentences))

# append vader_scores 
tweets_merged['compound'] = tweets_merged['vader_scores'].apply(lambda score_dict: score_dict['compound'])

# check data
tweets_merged.head()


# In[39]:


#  === Engagement Score ===
# creating a sensible engagement score through normalization
x = tweets_merged[['comment_num','retweet_num','like_num']].values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
engagement_score = df[0]+df[1]+df[2]

# === Approval Score ===
x2 = tweets_merged[['retweet_num','like_num']].values 
x2_scaled = min_max_scaler.fit_transform(x2)
df2 = pd.DataFrame(x2_scaled)
approval_score = df2[0]+df2[1]

# === Controversial Score === 
x3 = tweets_merged[['comment_num']].values 
x3_scaled = min_max_scaler.fit_transform(x3)
df3 = pd.DataFrame(x3_scaled)
controversial_score = df3[0]

# append engagement_score & compound_engagement_score
tweets_merged['engagement_score'] = engagement_score
tweets_merged['approval_score'] = approval_score
tweets_merged['controversial_score'] = controversial_score

# creating the combined compound metric
tweets_merged['compound_engagement_score'] = tweets_merged['engagement_score']*tweets_merged['compound']
tweets_merged['compound_approval_score'] = tweets_merged['approval_score']*tweets_merged['compound']
tweets_merged['compound_controversial_score'] = tweets_merged['controversial_score']*tweets_merged['compound']
tweets_merged.head()


# # Data Analysis

# ### Apple Analysis

# In[40]:


# group sentiment by date - APPLE ONLY
tweets_merged_grouped_apple_a = tweets_merged[tweets_merged['ticker_symbol']=='AAPL'].groupby(pd.to_datetime(tweets_merged['date_s']).dt.date).mean()

# convert to datetime
apple_p.date_s = pd.to_datetime(apple_p.date_s)
tweets_merged_grouped_apple_a.index = pd.to_datetime(tweets_merged_grouped_apple_a.index)

# merge the two datasets
tweets_merged_grouped_apple = pd.merge(tweets_merged_grouped_apple_a, apple_p, on='date_s', how='inner')


# In[41]:


# create column which contains the delta of the compound
tweets_merged_grouped_apple["delta_compound"] = tweets_merged_grouped_apple["compound"].pct_change()

# check data 
tweets_merged_grouped_apple.head()


# > General idea: % ∆ Sentiment => % ∆ Returns 

# In[42]:


# plot tweets_merged_grouped 
fig,ax = plt.subplots(figsize = (10, 5))
ax.plot(tweets_merged_grouped_apple.date_s, 
        tweets_merged_grouped_apple['compound'].rolling(window=30).mean(),
        color = 'royalblue')
plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
plt.xticks(rotation=45)
ax2 = ax.twinx()
ax2.plot(tweets_merged_grouped_apple['date_s'], 
            tweets_merged_grouped_apple['Close'].rolling(window=30).mean(),
            color = "orange")
# ax.set_xlabel('Date', fontsize=16)
ax.set_ylabel('Compound Score', color = 'royalblue', fontsize=14)
ax2.set_ylabel('Close ($)', color = 'orange', fontsize=14)
plt.title("Apple", fontsize=18, pad='15.0')

plt.savefig("Apple.png", dpi=460)


# In[43]:


# check whether all variables are distributed normally

l = list(tweets_merged_grouped_apple.columns.values)[1:]
counter = -1
 
for i in l:
    x = stats.normaltest(tweets_merged_grouped_apple[i][1:])
    counter = counter + 1
    if x.pvalue < 0.05: 
        print(l[counter],x.pvalue.round(3), 'TRUE')
    else: 
        print(l[counter],x.pvalue.round(3), 'FALSE')


# In[44]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_apple["delta_compound"][1:], tweets_merged_grouped_apple["daily_returns"][1:])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value.round(3))


# In[45]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_apple["compound"], tweets_merged_grouped_apple["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# In[46]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_apple["compound_engagement_score"], tweets_merged_grouped_apple["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# In[47]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_apple["compound_approval_score"], tweets_merged_grouped_apple["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# In[48]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_apple["compound_controversial_score"], tweets_merged_grouped_apple["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# #### Check for Lag 
# Does the sentiment from the previous day predictive power for the next day's daily return?

# In[49]:


# check for lag

# apple delta return (has to be one day ahead )
ap_dr = tweets_merged_grouped_apple["Close"].shift(2)[3:]
# apple delta compound 
ap_dc = tweets_merged_grouped_apple["compound"][3:]


# correlation analysis
corr, p_value = scipy.stats.pearsonr(ap_dc, ap_dr)

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value.round(3))


# #### Stationary Check
# 
# Augmented Dickey-Fuller test
# Statistical tests make strong assumptions about your data. They can only be used to inform the degree to which a null hypothesis can be rejected or fail to be reject. The result must be interpreted for a given problem to be meaningful.
# 
# Nevertheless, they can provide a quick check and confirmatory evidence that your time series is stationary or non-stationary.
# 
# The Augmented Dickey-Fuller test is a type of statistical test called a unit root test.
# 
# The intuition behind a unit root test is that it determines how strongly a time series is defined by a trend.
# 
# There are a number of unit root tests and the Augmented Dickey-Fuller may be one of the more widely used. It uses an autoregressive model and optimizes an information criterion across multiple different lag values.
# 
# The null hypothesis of the test is that the time series can be represented by a unit root, that it is not stationary (has some time-dependent structure). The alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary.
# 
# Null Hypothesis (H0): If failed to be rejected, it suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure.
# Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure.
# We interpret this result using the p-value from the test. A p-value below a threshold (such as 5% or 1%) suggests we reject the null hypothesis (stationary), otherwise a p-value above the threshold suggests we fail to reject the null hypothesis (non-stationary).
# 
# p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
# p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

# In[50]:


# create stationary time series
stationary_close = tweets_merged_grouped_apple[["Close"]].pct_change().dropna().values
stationary_compound = tweets_merged_grouped_apple[["compound"]].pct_change().dropna().values

# print results for close
result_close = adfuller(stationary_close)
print('ADF Statistic: %f' % result_close[0])
print('p-value: %f' % result_close[1])
print('Critical Values:')
for key, value in result_close[4].items():
	print('\t%s: %.3f' % (key, value))
    
# print results for compound
result_compound = adfuller(stationary_compound)
print('ADF Statistic: %f' % result_compound[0])
print('p-value: %f' % result_compound[1])
print('Critical Values:')
for key, value in result_compound[4].items():
	print('\t%s: %.3f' % (key, value))


# #### Ganger Causality Analysis 
# The Null hypothesis for grangercausalitytests is that __the time series in the second column, x2, does NOT Granger cause the time series in the first column, x1__. Grange causality means that past values of x2 have a statistically significant effect on the current value of x1, taking past values of x1 into account as regressors. We reject the null hypothesis that x2 does not Granger cause x1 if the pvalues are below a desired size of the test.

# In[51]:


# create data 
data = tweets_merged_grouped_apple[["Close", "compound"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# > For number of lags (no zero) __1__ when p-value shows up as p=0.4037, this means that past __1__ values of __"compound"__ (jointly) __have no statistically significant effect__ on the current value of the __"Close"__ column (first column).
# 
# 
# > For number of lags (no zero) __14__ when p-value shows up as p=0.0003, this means that past __14__ values of "compound" (jointly) __have statistically significant effect__ on the current value of the "Close" column (first column).

# In[52]:


# create data 
data = tweets_merged_grouped_apple[["compound", "Close"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# > For number of lags (no zero) __1__ when p-value shows up as p=0.0085, this means that past __1__ values of __"Close"__ (jointly) __have statistically significant effect__ on the current value of the __"compound"__ column (first column).

# ### Amazon Analysis

# In[53]:


# group sentiment by date - APPLE ONLY
tweets_merged_grouped_amazon_a = tweets_merged[tweets_merged['ticker_symbol']=='AMZN'].groupby(pd.to_datetime(tweets_merged['date_s']).dt.date).mean()

# convert to datetime
amazon_p.date_s = pd.to_datetime(amazon_p.date_s)
tweets_merged_grouped_amazon_a.index = pd.to_datetime(tweets_merged_grouped_amazon_a.index)

# merge the two datasets
tweets_merged_grouped_amazon = pd.merge(tweets_merged_grouped_amazon_a, amazon_p, on='date_s', how='inner')


# In[54]:


# create column which contains the delta 
tweets_merged_grouped_amazon["delta_compound"] = tweets_merged_grouped_amazon["compound"].pct_change()
# check data 
tweets_merged_grouped_amazon.head()


# In[55]:


# plot tweets_merged_grouped 
fig,ax = plt.subplots(figsize = (10, 5))
ax.plot(tweets_merged_grouped_amazon.date_s, 
        tweets_merged_grouped_amazon['compound'].rolling(window=30).mean(),
        color = 'royalblue')
plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
plt.xticks(rotation=45)
ax2 = ax.twinx()
ax2.plot(tweets_merged_grouped_amazon['date_s'], 
            tweets_merged_grouped_amazon['Close'].rolling(window=30).mean(),
            color = "orange")
# ax.set_xlabel('Date', fontsize=16)
ax.set_ylabel('Compound Score', color = 'royalblue', fontsize=14)
ax2.set_ylabel('Close ($)', color = 'orange', fontsize=14)
plt.title("Amazon", fontsize=18, pad='15.0')

plt.savefig("Amazon.png", dpi=460)


# In[56]:


# check whether all variables are distributed normally

l = list(tweets_merged_grouped_amazon.columns.values)[1:]
counter = -1

for i in l:
    x = stats.normaltest(tweets_merged_grouped_amazon[i][1:])
    counter = counter + 1
    if x.pvalue < 0.05: 
        print(l[counter],x.pvalue.round(3), 'TRUE')
    else: 
        print(l[counter],x.pvalue.round(3), 'FALSE')


# In[57]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_amazon["delta_compound"][1:], tweets_merged_grouped_amazon["daily_returns"][1:])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value.round(3))


# In[58]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_amazon["compound"], tweets_merged_grouped_amazon["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# In[59]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_amazon["compound_engagement_score"], tweets_merged_grouped_amazon["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# In[60]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_amazon["compound_approval_score"], tweets_merged_grouped_amazon["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# In[61]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_amazon["compound_controversial_score"], tweets_merged_grouped_amazon["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# #### Check for Lag 
# Does the sentiment from the previous day predictive power for the next day's daily return?

# In[62]:


# check for lag

# amazon delta return (has to be one day ahead )
az_dr = tweets_merged_grouped_amazon["Close"].shift(2)[3:]
# apple delta compound 
az_dc = tweets_merged_grouped_amazon["compound"][3:]


# correlation analysis
corr, p_value = scipy.stats.pearsonr(az_dc, az_dr)

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value.round(3))


# #### Stationary Check

# In[63]:


# create stationary time series
stationary_close = tweets_merged_grouped_amazon[["Close"]].pct_change().dropna().values
stationary_compound = tweets_merged_grouped_amazon[["compound"]].pct_change().dropna().values

# print results for close
result_close = adfuller(stationary_close)
print('ADF Statistic: %f' % result_close[0])
print('p-value: %f' % result_close[1])
print('Critical Values:')
for key, value in result_close[4].items():
	print('\t%s: %.3f' % (key, value))
    
# print results for compound
result_compound = adfuller(stationary_compound)
print('ADF Statistic: %f' % result_compound[0])
print('p-value: %f' % result_compound[1])
print('Critical Values:')
for key, value in result_compound[4].items():
	print('\t%s: %.3f' % (key, value))


# #### Ganger Causality Analysis 
# The Null hypothesis for grangercausalitytests is that the time series in the second column, x2, does NOT Granger cause the time series in the first column, x1. Grange causality means that past values of x2 have a statistically significant effect on the current value of x1, taking past values of x1 into account as regressors. We reject the null hypothesis that x2 does not Granger cause x1 if the pvalues are below a desired size of the test.

# In[64]:


# create data 
data = tweets_merged_grouped_amazon[["Close", "compound"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# > For number of lags (no zero) __1__ when p-value shows up as p=0.0262, this means that past __1__ values of __"compound"__ (jointly) __have a statistically significant effect__ on the current value of the __"Close"__ column (first column).

# In[65]:


# create data 
data = tweets_merged_grouped_amazon[["compound", "Close"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# > For number of lags (no zero) __3__ when p-value shows up as p=0.1750, this means that past __3__ values of __"Close"__ (jointly) __have no statistically significant effect__ on the current value of the __"compound"__ column (first column).

# ### Microsoft Analysis

# In[66]:


# group sentiment by date - APPLE ONLY
tweets_merged_grouped_microsoft_a = tweets_merged[tweets_merged['ticker_symbol']=='MSFT'].groupby(pd.to_datetime(tweets_merged['date_s']).dt.date).mean()

# convert to datetime
microsoft_p.date_s = pd.to_datetime(microsoft_p.date_s)
tweets_merged_grouped_microsoft_a.index = pd.to_datetime(tweets_merged_grouped_microsoft_a.index)

# merge the two datasets
tweets_merged_grouped_microsoft = pd.merge(tweets_merged_grouped_microsoft_a, microsoft_p, on='date_s', how='inner')


# In[67]:


# create column which contains the delta 
tweets_merged_grouped_microsoft["delta_compound"] = tweets_merged_grouped_microsoft["compound"].pct_change()

# check data 
tweets_merged_grouped_microsoft.head()


# In[68]:


# plot tweets_merged_grouped 
fig,ax = plt.subplots(figsize = (10, 5))
ax.plot(tweets_merged_grouped_microsoft.date_s, 
        tweets_merged_grouped_microsoft['compound'].rolling(window=30).mean(),
        color = 'royalblue')
plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
plt.xticks(rotation=45)
ax2 = ax.twinx()
ax2.plot(tweets_merged_grouped_microsoft['date_s'], 
            tweets_merged_grouped_microsoft['Close'].rolling(window=30).mean(),
            color = "orange")
# ax.set_xlabel('Date', fontsize=16)
ax.set_ylabel('Compound Score', color = 'royalblue', fontsize=14)
ax2.set_ylabel('Close ($)', color = 'orange', fontsize=14)
plt.title("Microsoft", fontsize=18, pad='15.0')

plt.savefig("Microsoft.png", dpi=460)


# In[69]:


# check whether all variables are distributed normally

l = list(tweets_merged_grouped_microsoft.columns.values)[1:]
counter = -1

for i in l:
    x = stats.normaltest(tweets_merged_grouped_microsoft[i][1:])
    counter = counter + 1
    if x.pvalue < 0.05: 
        print(l[counter],x.pvalue.round(3), 'TRUE')
    else: 
        print(l[counter],x.pvalue.round(3), 'FALSE')


# In[70]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_microsoft["delta_compound"][1:], tweets_merged_grouped_microsoft["daily_returns"][1:])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value.round(3))


# In[71]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_microsoft["compound"], tweets_merged_grouped_microsoft["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# In[72]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_microsoft["compound_engagement_score"], tweets_merged_grouped_microsoft["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# In[73]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_microsoft["compound_approval_score"], tweets_merged_grouped_microsoft["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# In[74]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_microsoft["compound_controversial_score"], tweets_merged_grouped_microsoft["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# #### Check for Lag 
# Does the sentiment from the previous day predictive power for the next day's daily return?

# In[75]:


# check for lag

# amazon delta return (has to be one day ahead )
az_dr = tweets_merged_grouped_microsoft["Close"].shift(2)[3:]
# apple delta compound 
az_dc = tweets_merged_grouped_microsoft["compound"][3:]


# correlation analysis
corr, p_value = scipy.stats.pearsonr(az_dc, az_dr)

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value.round(3))


# #### Stationary Check

# In[76]:


# create stationary time series
stationary_close = tweets_merged_grouped_microsoft[["Close"]].pct_change().dropna().values
stationary_compound = tweets_merged_grouped_microsoft[["compound"]].pct_change().dropna().values

# print results for close
result_close = adfuller(stationary_close)
print('ADF Statistic: %f' % result_close[0])
print('p-value: %f' % result_close[1])
print('Critical Values:')
for key, value in result_close[4].items():
	print('\t%s: %.3f' % (key, value))
    
# print results for compound
result_compound = adfuller(stationary_compound)
print('ADF Statistic: %f' % result_compound[0])
print('p-value: %f' % result_compound[1])
print('Critical Values:')
for key, value in result_compound[4].items():
	print('\t%s: %.3f' % (key, value))


# #### Ganger Causality Analysis 
# The Null hypothesis for grangercausalitytests is that the time series in the second column, x2, does NOT Granger cause the time series in the first column, x1. Grange causality means that past values of x2 have a statistically significant effect on the current value of x1, taking past values of x1 into account as regressors. We reject the null hypothesis that x2 does not Granger cause x1 if the pvalues are below a desired size of the test.

# In[77]:


# create data 
data = tweets_merged_grouped_microsoft[["Close", "compound"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# > For number of lags (no zero) __1__ when p-value shows up as p=0.3589, this means that past __1__ values of __"compound"__ (jointly) __have no statistically significant effect__ on the current value of the __"Close"__ column (first column).

# In[78]:


# create data 
data = tweets_merged_grouped_microsoft[["compound", "Close"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# > For number of lags (no zero) __1__ when p-value shows up as p=0.4402, this means that past __1__ values of __"Close"__ (jointly) __have no statistically significant effect__ on the current value of the __"compound"__ column (first column).

# ### Tesla Analysis

# In[79]:


# group sentiment by date - APPLE ONLY
tweets_merged_grouped_tesla_a = tweets_merged[tweets_merged['ticker_symbol']=='TSLA'].groupby(pd.to_datetime(tweets_merged['date_s']).dt.date).mean()

# convert to datetime
tesla_p.date_s = pd.to_datetime(tesla_p.date_s)
tweets_merged_grouped_tesla_a.index = pd.to_datetime(tweets_merged_grouped_tesla_a.index)

# merge the two datasets
tweets_merged_grouped_tesla = pd.merge(tweets_merged_grouped_tesla_a, tesla_p, on='date_s', how='inner')


# In[80]:


# create column which contains the delta 
tweets_merged_grouped_tesla["delta_compound"] = tweets_merged_grouped_tesla["compound"].pct_change()

# check data 
tweets_merged_grouped_tesla.head()


# In[81]:


# plot tweets_merged_grouped 
fig,ax = plt.subplots(figsize = (10, 5))
ax.plot(tweets_merged_grouped_tesla.date_s, 
        tweets_merged_grouped_tesla['compound'].rolling(window=30).mean(),
        color = 'royalblue')
plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
plt.xticks(rotation=45)
ax2 = ax.twinx()
ax2.plot(tweets_merged_grouped_tesla['date_s'], 
            tweets_merged_grouped_tesla['Close'].rolling(window=30).mean(),
            color = "orange")
# ax.set_xlabel('Date', fontsize=16)
ax.set_ylabel('Compound Score', color = 'royalblue', fontsize=14)
ax2.set_ylabel('Close ($)', color = 'orange', fontsize=14)
plt.title("Tesla", fontsize=18, pad='15.0')

plt.savefig("Tesla.png", dpi=460)


# In[82]:


# check whether all variables are distributed normally
l = list(tweets_merged_grouped_tesla.columns.values)[1:]
counter = -1

for i in l:
    x = stats.normaltest(tweets_merged_grouped_tesla[i][1:])
    counter = counter + 1
    if x.pvalue < 0.05: 
        print(l[counter],x.pvalue.round(3), 'TRUE')
    else: 
        print(l[counter],x.pvalue.round(3), 'FALSE')


# In[83]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_tesla["delta_compound"][1:], tweets_merged_grouped_tesla["daily_returns"][1:])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value.round(3))


# In[84]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_tesla["compound"], tweets_merged_grouped_tesla["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# In[85]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_tesla["compound_engagement_score"], tweets_merged_grouped_tesla["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# In[86]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_tesla["compound_approval_score"], tweets_merged_grouped_tesla["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# In[87]:


# correlation analysis 
corr, p_value = scipy.stats.pearsonr(tweets_merged_grouped_tesla["compound_controversial_score"], tweets_merged_grouped_tesla["Close"])

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value)


# #### Check for Lag 
# Does the sentiment from the previous day have predictive power for the next day's daily return?

# In[88]:


# check for lag

# amazon delta return (has to be one day ahead )
az_dr = tweets_merged_grouped_tesla["Close"].shift(2)[3:]
# apple delta compound 
az_dc = tweets_merged_grouped_tesla["compound"][3:]


# correlation analysis
corr, p_value = scipy.stats.pearsonr(az_dc, az_dr)

print('Correlation Coefficient:',corr.round(3))
print('P-Value:',p_value.round(3))


# #### Stationary Check

# In[89]:


# create stationary time series
stationary_close = tweets_merged_grouped_tesla[["Close"]].pct_change().dropna().values
stationary_compound = tweets_merged_grouped_tesla[["compound"]].pct_change().dropna().values

# print results for close
result_close = adfuller(stationary_close)
print('ADF Statistic: %f' % result_close[0])
print('p-value: %f' % result_close[1])
print('Critical Values:')
for key, value in result_close[4].items():
	print('\t%s: %.3f' % (key, value))
    
# print results for compound
result_compound = adfuller(stationary_compound)
print('ADF Statistic: %f' % result_compound[0])
print('p-value: %f' % result_compound[1])
print('Critical Values:')
for key, value in result_compound[4].items():
	print('\t%s: %.3f' % (key, value))


# #### Ganger Causality Analysis 
# The Null hypothesis for grangercausalitytests is that the time series in the second column, x2, does NOT Granger cause the time series in the first column, x1. Grange causality means that past values of x2 have a statistically significant effect on the current value of x1, taking past values of x1 into account as regressors. We reject the null hypothesis that x2 does not Granger cause x1 if the pvalues are below a desired size of the test.
# 
# > whether the time series in the second column Granger causes the time series in the first column.

# In[90]:


# create data 
data = tweets_merged_grouped_tesla[["Close", "compound"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# > For number of lags (no zero) __1__ when p-value shows up as p=0.6244, this means that past __1__ values of __"compound"__ (jointly) __have no statistically significant effect__ on the current value of the __"Close"__ column (first column).

# In[91]:


# create data 
data = tweets_merged_grouped_tesla[["compound", "Close"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# > For number of lags (no zero) __17__ when p-value shows up as p=0.1674, this means that past __17__ values of __"Close"__ (jointly) __have no statistically significant effect__ on the current value of the __"compound"__ column (first column).

# # Simulation

# In[92]:


# Import Stock Data for Testing Period

# apple stock data
sim_apple_p = yahoo('AAPL','2019-01-01','2019-12-31','apple_p')

# amazon stock data 
sim_amazon_p = yahoo('AMZN','2019-01-01','2019-12-31','amazon_p')

# microsoft stock data 
sim_microsoft_p = yahoo('MSFT','2019-01-01','2019-12-31','microsoft_p')

# tesla stock data 
sim_tesla_p = yahoo('TSLA','2019-01-01','2019-12-31','tesla_p')


# In[93]:


def grouped_df_simulation(sim_tweets_df, sim_stock_df):
    # merge the two datasets
    df = pd.merge(sim_tweets_df, sim_stock_df, on='date_s', how='inner')

    # calculate percentage change 
    df['compound_pct'] = df['compound'].pct_change()

    # calculate percentage change 
    df['close_pct'] = df['Close'].pct_change()
    
    return df


# In[94]:


# create rolling dataframe
def create_rolling_df(df, window, df1):
    
    # define benchmark one std of previous mean
    benchmark = df1['compound'].rolling(window=window).mean().std()
    
    # define benchmark mean of previous rolling_pct_change
    benchmark2 = df1['compound'].rolling(window=window).mean().pct_change().mean()
    
    # define benchmark mean of previous compound
    benchmark3 = df1['compound'].rolling(window=window).mean().mean()
    
    
    
    
    # create rolling average column 
    rolling_compound = df['compound'].rolling(window=window).mean()
    # create percentage change of rolling average column 
    rolling_pct_change = rolling_compound.pct_change()
    # append the new columns
    df['rolling_compound'] = rolling_compound
    df['rolling_pct_change'] = rolling_pct_change
    
    # initilize signal list 
    signal = []

    for i, value in enumerate(df['rolling_pct_change']): 
        if value >= 0:
            signal.append('Buy')
        elif value < 0: 
            signal.append('Sell')
    # since rolling_pct_change has NA's we need one neutral signal for those rows in order to have the same length
        else:
            signal.append('/')
    
    df['signal'] = signal
    
    return df


# In[95]:


# brute force version
# only considers buying signals

def simulation(df):
    strategy = 100
    
    # buy_hold means we buy and hold the stock in the given period
    buy_hold = ((df['Close'][len(df)-1]/df['Close'][0]))*100

    for num, value in enumerate(df['signal']): 
        # num has to be smaller than the length of the df -1 because of num + 1 in the if clause 
        if value == 'Buy' and num < len(df)-1:
            # we get the percentage change if we buy 
            # when the signal is Sell, then nothing happens to our Portfolio since we are not owning the stock anymore
            strategy = strategy * (1+df['close_pct'][num+1])
    return print("Sentiment Strategy:", strategy, "\n", "Buy And Hold:", buy_hold)


# In[96]:


def sim_visualization(df,title):
    buy_signal = []
    sell_signal = []
    hold_last_signal = []

    for i in range(len(df)):
        if df['signal'][i] == 'Buy':
            buy_signal.append(df['Close'][i])
            sell_signal.append(float('nan'))
            hold_last_signal.append(float('nan'))
        elif df['signal'][i] == 'Sell':
            sell_signal.append(df['Close'][i])
            buy_signal.append(float('nan'))
            hold_last_signal.append(float('nan'))
        else:
            hold_last_signal.append(df['Close'][i])
            buy_signal.append(float('nan'))
            sell_signal.append(float('nan'))


    df['buy_signal'] = buy_signal
    df['sell_signal'] = sell_signal
    df['hold_last_signal'] = hold_last_signal

        # example plot
    fig = plt.subplots(figsize = (20, 10))
    plt.plot(df['Close'], alpha = 0.5)
    plt.scatter(df.index, df['buy_signal'], marker='^', color='green')
    plt.scatter(df.index, df['sell_signal'], marker='v', color='red')
    plt.scatter(df.index, df['hold_last_signal'], marker='D', color='blue')
    plt.title("{} Simulation".format(title))
    plt.show()


# ## Apple Simulation

# In[97]:


# group df
apple_grouped_sim = grouped_df_simulation(tweets_merged_grouped_apple_a, sim_apple_p)

# define lag
# 12, 13, 14, 15, 16, 17
lag = 17

# create signals and other new columns 
apple_grouped_sim = create_rolling_df(apple_grouped_sim, lag, tweets_merged_grouped_apple_a)
apple_grouped_sim = apple_grouped_sim.reset_index()

# simulate the strategy based on the signals 
simulation(apple_grouped_sim)


# In[98]:


# visualize simulation
# sim_visualization(apple_grouped_sim, 'Apple')


# ##  Amazon Simulation

# In[99]:


# group df
amazon_grouped_sim = grouped_df_simulation(tweets_merged_grouped_amazon_a, sim_amazon_p)

# define lag
lag = 15

# create signals and other new columns 
amazon_grouped_sim = create_rolling_df(amazon_grouped_sim, lag,tweets_merged_grouped_amazon_a)
amazon_grouped_sim = amazon_grouped_sim.reset_index()

# simulate the strategy based on the signals 
simulation(amazon_grouped_sim)


# In[100]:


# visualize simulation
# sim_visualization(amazon_grouped_sim, 'Amazon')


# ##  Microsoft Simulation

# In[101]:


# group df
microsoft_grouped_sim = grouped_df_simulation(tweets_merged_grouped_microsoft_a, sim_microsoft_p)

# define lag
lag = 5

# create signals and other new columns 
microsoft_grouped_sim = create_rolling_df(microsoft_grouped_sim, lag, tweets_merged_grouped_microsoft_a)
microsoft_grouped_sim = microsoft_grouped_sim.reset_index()

# simulate the strategy based on the signals 
simulation(microsoft_grouped_sim)


# ##  Tesla Simulation

# In[102]:


# group df
tesla_grouped_sim = grouped_df_simulation(tweets_merged_grouped_tesla_a, sim_tesla_p)

# define lag
lag = 4

# create signals and other new columns 
tesla_grouped_sim = create_rolling_df(tesla_grouped_sim, lag, tweets_merged_grouped_tesla_a)
tesla_grouped_sim = tesla_grouped_sim.reset_index()

# simulate the strategy based on the signals 
simulation(tesla_grouped_sim)


# # Validating
# 
# > __Why do the simulations perform so badly?__
# >> Hypothesis: The pattern that were detected previously did not withstand into the new period.

# ## Apple

# In[103]:


# create data 
data = apple_grouped_sim[["Close", "compound"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# In[104]:


# create data 
data = apple_grouped_sim[["compound", "Close"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# ## Amazon 

# In[105]:


# create data 
data = amazon_grouped_sim[["Close", "compound"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# In[106]:


# create data 
data = amazon_grouped_sim[["compound", "Close"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# ## Microsoft

# In[107]:


# create data 
data = microsoft_grouped_sim[["Close", "compound"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# In[108]:


# create data 
data = microsoft_grouped_sim[["compound", "Close"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# ## Tesla 

# In[109]:


# create data 
data = tesla_grouped_sim[["Close", "compound"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# In[110]:


# create data 
data = tesla_grouped_sim[["compound", "Close"]].pct_change().dropna()
# execute granger causality test
gc_res = grangercausalitytests(data, 20)


# # The Following Content was not Included in the Bachelor Thesis Results
# 
# ***

# # Linear Regression

# In[111]:


x = np.array(tweets_merged_grouped_apple.compound).reshape((-1, 1))
y = np.array(tweets_merged_grouped_apple.Close)

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)


x2 = np.array(apple_grouped_sim.compound).reshape((-1, 1))
y_pred = model.predict(x2)

plt.plot(y_pred)
plt.title('Prediction')
plt.show()

plt.plot(apple_grouped_sim.Close)
plt.title('Actual')
plt.show()


# # Vector Autoregression
# 
# 1. You need at least two time series (variables)
# 2. The time series should influence each other.
# 
# It is considered as an Autoregressive model because, each variable (Time Series) is modeled as a function of the past values, that is the predictors are nothing but the lags (time delayed value) of the series.
# 
# 
# Ok, so how is VAR different from other Autoregressive models like AR, ARMA or ARIMA?
# 
# The primary difference is those models are uni-directional, where, the predictors influence the Y and not vice-versa. Whereas, Vector Auto Regression (VAR) is bi-directional. That is, the variables influence each other.

# In[112]:


# split into test and training data set
# tweets_merged_grouped_apple.set_index('date_s', inplace=True)
train_data_apple = tweets_merged_grouped_apple[['compound','Close']].astype('float').pct_change().dropna()

# apple_grouped_sim.set_index('date_s', inplace=True)
test_data_apple = apple_grouped_sim[['compound','Close']].astype('float').pct_change().dropna()

print(test_data_apple.shape, train_data_apple.shape)
print(test_data_apple, train_data_apple)


# In[113]:


# --- VARMAX --- 

# Train

# 12 = the number of lags; 0 means no moving averages; 
var_max_model = VARMAX(train_data_apple, order=(12,0))
fitted_model = var_max_model.fit()
print(fitted_model.summary())


# In[114]:


# Test
n_forecast = 50
predict = fitted_model.get_prediction()

predictions = predict.predicted_mean
predictions.columns = ['compound','Close']


predictions

# test_vs_pred = pd.concat([test_data_apple,predictions],axis=1)

# test_vs_pred.plot(figsize=(12,5))
# predictions.plot()


# In[115]:


# --- VAR ---

model = VAR(train_data_apple)
model_fit = model.fit(maxlags=17)
model_fit.summary()

# Forecast

# derive lag from model_fit
lag_order = model_fit.k_ar

# model input 
forecast_input = np.array(test_data_apple.values[-lag_order:])

# build the forecast
fc = model_fit.forecast(forecast_input, steps=12)

# plot forecast
model_fit.plot_forecast(100)

