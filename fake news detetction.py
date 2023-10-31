#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[5]:


fake_data=pd.read_csv("Downloads/Fake.csv")
true_data=pd.read_csv("Downloads/True.csv")


# In[6]:


fake_data["class"]=0
true_data["class"]=1


# In[7]:


fake_data.shape[0],true_data.shape[0]


# In[ ]:


fake_data_manual_test=fake_data.tail(10)
for i in range(23480,23470,-1):
    fake_data.drop([i],axis=0,inplace=True)

true_data_manual_test=true_data.tail(10)
for i in range(21416,21406,-1):
    true_data.drop([i],axis=0,inplace=True)


# In[9]:


fake_data.shape[0],true_data.shape[0]


# In[ ]:


fake_data_manual_test["class"]=0
true_data_manual_test["class"]=1


# In[12]:


fake_data_manual_test.head()
true_data_manual_test.head()


# In[14]:


data_merge=pd.concat([fake_data,true_data],axis=0)
data_merge.head(10)


# In[18]:


data_merge.columns


# In[19]:


data=data_merge.drop(['title','subject','date'],axis=1)


# In[21]:


data.isnull().sum()


# In[22]:


data=data.sample(frac=1)


# In[23]:


data.head()


# In[24]:


data.reset_index(inplace=True)


# In[25]:


data.head()


# In[26]:


data.drop(['index'],axis=1,inplace=True)


# In[27]:


data.head()


# In[32]:


def modtxt(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\\W"," ",text)
    text=re.sub('https?://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]'% re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    return text


# In[33]:


data['text']=data['text'].apply(modtxt)


# In[34]:


x=data['text']
y=data['class']


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[42]:


from sklearn.feature_extraction.text import TfidfVectorizer
vtr=TfidfVectorizer()
xv_train=vtr.fit_transform(x_train)
xv_test=vtr.transform(x_test)


# In[45]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(xv_train,y_train)


# In[46]:


pred=LR.predict(xv_test)


# In[47]:


LR.score(xv_test,y_test)


# In[50]:


print(classification_report(y_test,pred))


# In[51]:


from sklearn.tree import DecisionTreeClassifier
d=DecisionTreeClassifier()
d.fit(xv_train,y_train)


# In[52]:


pred_dt=d.predict(xv_test)


# In[54]:


d.score(xv_test,y_test)


# In[55]:


print(classification_report(y_test,pred_dt))


# In[58]:


from sklearn.ensemble import GradientBoostingClassifier

gb=GradientBoostingClassifier(random_state=0)
gb.fit(xv_train,y_train)


# In[59]:


pred_gb=gb.predict(xv_test)


# In[60]:


gb.score(xv_test,y_test)


# In[61]:


print(classification_report(y_test,pred_gb))


# In[62]:


from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(random_state=0)
rf.fit(xv_train,y_train)


# In[64]:


pred_rf=rf.predict(xv_test)


# In[65]:


rf.score(xv_test,y_test)


# In[66]:


print(classification_report(y_test,pred_rf))


# In[98]:


def out_ans(n):
    if n==0:
        return "fake news"
    elif n==1:
        return "correct news"

def testing(news):
    test_news={"text":[news]}
    new_news=pd.DataFrame(test_news)
    new_news['text']=new_news['text'].apply(modtxt)
    news_x_test=new_news['text']
    xv=vtr.transform(news_x_test)
    pred_lr=LR.predict(xv)
    pred_dt=d.predict(xv)
    pred_gb=gb.predict(xv)
    pred_rf=rf.predict(xv)
    print("lr:{0} dt:{1} gb:{2} rf:{3}".format(out_ans(pred_lr[0]),out_ans(pred_dt[0]),out_ans(pred_gb[0]),out_ans(pred_rf[0])))


# In[99]:


news=str(input())
testing(news)

