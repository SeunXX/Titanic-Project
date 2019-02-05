
# coding: utf-8

# In[198]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree


# In[199]:


# Use training data to first create a model- classification model: decision Tree using gini criteria as classifier

titanic_train_db = pd.read_csv('C:/Users/Seun/Desktop/Py_ML_Data/Titanic/train.csv')


# In[201]:


titanic_train_db


# In[202]:


len(titanic_train_db)


# In[203]:


titanic_train_db.shape


# In[204]:


titanic_train_db.columns


# In[205]:


titanic_train_db.head(10)


# In[206]:


#predictor_variable: Pclass = ticket class, Fare = Amount passenger paid, Parch = # of parents / children aboard the Titanic
All_features = ['PassengerId', 'Name', 'Pclass','Fare', 'Parch']
features = ['PassengerId','Pclass','Fare', 'Parch']
X = titanic_train_db[features]


# In[207]:


type(X['Pclass'])


# In[208]:


#outcome variable that we want to predict
Y = titanic_train_db.loc[:, 'Survived']


# In[209]:


#Split larger training dataset into test and training
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# In[210]:


X_test


# In[211]:


#Train the model
Titanic_gini = DecisionTreeClassifier(criterion = 'gini', random_state = 100,
                               max_depth=3, min_samples_leaf=5)


# In[212]:


Titanic_gini.fit(X_train, y_train)


# In[ ]:


#Predicting the outcome variable using the subdata X_test


# In[224]:


y_prediction = Titanic_gini.predict(X_test)
y_prediction 


# In[214]:


len(y_prediction)


# In[215]:


y_predictionDF = pd.DataFrame(data = y_prediction, columns = ['survived'])
y_predictionDF


# In[239]:


RightDF = X_test
RightDF


# In[241]:


y_DF = RightDF.merge(y_predictionDF, left_on = 'PassengerId' , right_on = 'survived')


# In[249]:


selection = ['PassengerId', 'survived']
y_DF = y_DF[selection]
y_DF


# In[192]:


#Accuracy of our model: the ratio of the predicted outcome to all the predicted data point
print('Accuracy is:')
accuracy = accuracy_score(y_test,y_prediction)*100
print(round(accuracy))


# In[436]:


Titanic_test_db = pd.read_csv('C:/Users/Seun/Desktop/Py_ML_Data/Titanic/test.csv')


# In[195]:


Titanic_test_db


# In[197]:


features = ['Pclass','Fare', 'Parch']
test_features = Titanic_test_db[features]


# In[432]:


#Defining predictor variable on the test data
X_Titanic_test_db = test_features
print(X_Titanic_test_db)


# In[437]:


X_Titanic_test_db.head()


# In[456]:


X_Titanic_test = X_Titanic_test_db.fillna(0)


# In[458]:


pd.isna(X_Titanic_test)


# In[459]:


y_Titanic_test_prediction = Titanic_gini.predict(X_Titanic_test)
y_Titanic_test_prediction


# In[477]:


Survived = y_Titanic_test_prediction
Test_features = ['PassengerId', 'Survived']
PassengerId


# In[479]:


Test_prediction = pd.DataFrame('PassengerId', 'Survived')

