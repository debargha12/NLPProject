#!/usr/bin/env python
# coding: utf-8

# In[11]:


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
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
import pickle


# In[12]:


f = open(sys.argv[1]) 
import pandas as pd
import json
data = json.load(f) 

    # Iterating through the json 
    # list 
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
df.to_pickle("./dataframe.pkl")
print("Data loaded to the dataframe")


# In[13]:


df = pd.read_pickle("./dataframe.pkl")
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


def clean(data):
    data = data.lower()
    data = re.sub('<[^<]+?>', '', data)
    data=re.sub('[!#?,.:";]', '', data)
    data=re.sub(r'[0-9]+', '', data)
    data= re.sub("i'm","i am",data)
    data = stemSentence(data)
    return data


# In[15]:
print("Preprocessing started")
df['body'] = df['body'].map(lambda x : clean(x))
print("Preprocessing completed")

# In[16]:




emotions = ["anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust","neutral"]
train, test = train_test_split(df, random_state=40, test_size=0.25, shuffle=True)

X_train= train['body']
X_test= test['body']
print(X_train.shape)
print(X_test.shape)


# In[17]:

print("NB")
NB= Pipeline([('tfidf', TfidfVectorizer(stop_words=stop_words)),('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))),])
for emotion in emotions:
    NB.fit(X_train, train[emotion])
    prediction = NB.predict(X_test)
    print('Test accuracy of :'+emotion)
    print(accuracy_score(test[emotion], prediction))


# In[18]:

print("SVC")
SVC = Pipeline([('tfidf', TfidfVectorizer(stop_words=stop_words)),('clf', OneVsRestClassifier(LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000), n_jobs=1)),])
for emotion in emotions: 
    SVC.fit(X_train, train[emotion])
    prediction = SVC.predict(X_test)
    print('Test accuracy of :'+emotion)
    print(accuracy_score(test[emotion], prediction))
# In[19]:


print("PAC")
pac = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(PassiveAggressiveClassifier(random_state=0))),
            ])

for emotion in emotions:
    pac.fit(X_train, train[emotion])
    prediction = pac.predict(X_test)
    print('Test accuracy of :'+emotion)
    print(accuracy_score(test[emotion], prediction))


# In[21]:


import pickle
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(SVC, file)

# In[ ]:




# In[22]:


print("Model Saved")


# In[ ]:




