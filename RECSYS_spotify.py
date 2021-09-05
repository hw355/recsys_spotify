#!/usr/bin/env python
# coding: utf-8

# # A Recommendation Engine Using Python for An Episode Available on Spotify Podcast

# In[1]:


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import pandas
import numpy as np
import datetime as dt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from ipynb.fs.full.Credentials import *


# ## Data Collection

# In[2]:


client_credentials_manager = SpotifyClientCredentials(client_id = Spotify_Client_ID, 
                                                      client_secret = Spotify_Client_Secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


# In[3]:


episode_results = sp.search(q = 'data',
                            type = 'episode',
                            market = 'US',
                            limit = 1,
                            offset = 0
                           )
print(json.dumps(episode_results, indent=2))


# In[4]:


def getEpisodes(queries, total):
    name = []
    description = []
    duration = []
    explicit = []
    language = []
    release_date = []
    url = []

    if total <= 50:
        limit = total
    else:
        limit = 50

    for query in queries:
        print(query)
        for i in range(0, total, limit):
            results = sp.search(q = query,
                                type = 'episode',
                                market = 'US',
                                limit = limit,
                                offset = i
                               )
            
            for i, t in enumerate(results['episodes']['items']):
                name.append(t['name'])
                description.append(t['description'])
                duration.append(t['duration_ms'])
                explicit.append(t['explicit'])
                language.append(t['language'])
                release_date.append(t['release_date'])
                url.append(t['external_urls']['spotify'])

    pd.set_option('display.max_colwidth', 0)

    dataframe = pd.DataFrame({'Name' : name,
                              'Description' : description,
                              'Explicit' : explicit,
                              'Language' : language,
                              'Duration (ms)' : duration,
                              'Release Date' : release_date,
                              'URL' : url,
                             }).drop_duplicates().reset_index(drop = True)
    
    return dataframe


# In[5]:


episodes_data = getEpisodes(['h', 'b', 'f', 'm'], 1000)
print(episodes_data.shape)
episodes_data


# ## Exploratory Data Analysis

# In[6]:


episodes_data.info()


# In[7]:


episodes_data['Days ago'] = (np.datetime64(dt.date.today()) - episodes_data['Release Date'].values.astype('datetime64[D]')).astype(int)
episodes_data['Name + Description'] = episodes_data['Name'] + ' ' + episodes_data['Description']
episodes_data


# ## Feature Extraction

# In[ ]:


column_trans = ColumnTransformer([('exp_category', OneHotEncoder(dtype='int'), ['Explicit']),
                                  ('lang_category', OneHotEncoder(dtype='int'), ['Language']),
                                  ('text', TfidfVectorizer(use_idf = True, stop_words = 'english'), 'Name + Description'),
                                  ('Duration_int', MinMaxScaler(),['Duration (ms)']),
                                  ('Days_ago_int', MinMaxScaler(),['Days ago']),
    
                                 ],
                                 transformer_weights={'exp_category': 1.0,
                                                        'lang_category': 1.0,
                                                        'text': 1.0,
                                                        'Duration_int': 0.5,
                                                        'Days_ago_int': 0.5
                                                       },
                                 remainder='drop')

token = column_trans.fit_transform(episodes_data).toarray()
token


# ## Cosine Similarity

# In[9]:


cos_sim = cosine_similarity(token)
cos_sim


# In[10]:


cos_sim.shape


# In[11]:


list(episodes_data['Name'])


# ## Get Recommendation

# In[16]:


def getRecommendation(name, number):
    index = episodes_data[episodes_data['Name'] == name].index.values[0]
    scores = list(enumerate(cos_sim[index]))
    sorted_scores = sorted(scores, key = lambda x:x[1], reverse = True)

    n = 0
    print('The ' + str(number) + ' most recommended episodes to ' + name + ' are:\n')
    for index, score in sorted_scores[1:]:
        name = episodes_data[episodes_data.index == index]['Name'].values[0]
        url = episodes_data[episodes_data.index == index]['URL'].values[0]
        print(n + 1, index, name, '\n', url)
        print()
        n += 1
        if n > (number - 1):
            break


# In[11]:


list(episodes_data['Name'])


# In[23]:


getRecommendation('A Broken System for Housing the Homeless', 10)


# In[24]:


getRecommendation('Healthy Relationships', 10)


# In[25]:


getRecommendation('How to Decide', 10)

