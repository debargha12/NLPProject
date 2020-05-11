#!/usr/bin/env python
# coding: utf-8

# In[2]:

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
#get_ipython().run_line_magic('matplotlib', 'inline')
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
import pickle
import json

# In[4]:


f = open(sys.argv[1]) 


data = json.load(f) 

   
    #print(data["fkrr36o"]["emotion"])
f.close() 
fl=[]
    #print(data["fkrr36o"])
for u,v in data.items():
    l=[]
    l.append(u)
    l.append(v["body"])
    l.append(1 if v["emotion"]["anger"]==True else 0)
    l.append(1 if v["emotion"]["anticipation"]==True else 0)
    l.append(1 if v["emotion"]["disgust"]==True else 0)
    l.append(1 if v["emotion"]["fear"]==True else 0)
    l.append(1 if v["emotion"]["joy"]==True else 0)
    l.append(1 if v["emotion"]["love"]==True else 0)
    l.append(1 if v["emotion"]["optimism"]==True else 0)
    l.append(1 if v["emotion"]["pessimism"]==True else 0)
    l.append(1 if v["emotion"]["sadness"]==True else 0)
    l.append(1 if v["emotion"]["surprise"]==True else 0)
    l.append(1 if v["emotion"]["trust"]==True else 0)
    l.append(1 if v["emotion"]["neutral"]==True else 0)
    fl.append(l)

df = pd.DataFrame(fl, columns=["id","body","anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust","neutral"])
df.to_pickle("./dataframetest.pkl")
print("Data loaded to the dataframe")


# In[5]:

df = pd.read_pickle("./dataframetest.pkl")
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
porter = PorterStemmer()
def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


# In[6]:


def clean(data):
    data = data.lower()
    data = re.sub('<[^<]+?>', '', data)
    data=re.sub('[!#?,.:";]', '', data)
    data=re.sub(r'[0-9]+', '', data)
    data= re.sub("i'm","i am",data)
    data = stemSentence(data)
    return data


# In[7]:

print("Preprocessing started")
df['body'] = df['body'].map(lambda x : clean(x))
print("Preprocessing completed")

# In[9]:


from sklearn.feature_extraction.text import CountVectorizer

emotions = ["anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust","neutral"]
X_test= df['body']

print("Number of datapoints: ",X_test.shape[0])


# In[18]:

print("Loading model")
with open("pickle_model.pkl", 'rb') as file:
    pickle_model = pickle.load(file)
    

for emotion in emotions:
    prediction = pickle_model.predict(X_test)
    print('Test accuracy of :'+emotion)
    print(accuracy_score(df[emotion], prediction))


# In[ ]:




