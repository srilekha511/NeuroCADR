#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np


# In[5]:


df = pd.read_csv("DrugID_Names_Symptoms_Genes_1.csv")
df = df.drop_duplicates()
#df


# In[6]:


epilepy_ID = 'D004827'
epilepy_reflex_ID = 'D020195'


# In[7]:


test_data = df[df.iloc[:,0].str.contains(epilepy_ID)]
affected_rows = test_data.index


# In[8]:


train_data = df.drop(affected_rows)
X_train = train_data.iloc[:,1:-1]
y_train = train_data.iloc[:,-1]

X_test = test_data.iloc[:, 1:-1]
y_test = test_data.iloc[:,-1]


# In[9]:


model = NearestNeighbors(n_neighbors=6)
model = model.fit(X_train, y_train)


# In[14]:


y_pred = model.kneighbors(X_test, return_distance=False)


# In[18]:


pred_drugs = y_train[y_pred[0,:]]


# In[ ]:


pred_drugs

