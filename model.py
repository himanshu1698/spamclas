#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle


# In[15]:


df= pd.read_csv(r"C:\Users\Himanshu\Desktop\Spam\spam.csv", encoding="latin-1")


# In[17]:


#Features and Labels

df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

# xtract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
#    
pickle.dump(cv, open('tranform.pkl', 'wb'))
#    
#    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))
    


# In[ ]:




