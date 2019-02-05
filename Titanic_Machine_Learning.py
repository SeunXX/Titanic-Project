import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

# Use training data to first create a model- classification model: decision Tree using gini criteria as classifier

titanic_train_db = pd.read_csv('C:/Users/Seun/Desktop/Py_ML_Data/Titanic/train.csv')
titanic_train_db

len(titanic_train_db)
titanic_train_db.shape

titanic_train_db.columns
titanic_train_db.head(10)

#predictor_variable: Pclass = ticket class, Fare = Amount passenger paid, Parch = # of parents / children aboard the Titanic
features = ['PassengerId','Pclass','Fare', 'Parch']
X = titanic_train_db[features]

#outcome variable that we want to predict
Y = titanic_train_db.loc[:, 'Survived']

#Split larger training dataset into test and training
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
X_test

#Train the model
Titanic_gini = DecisionTreeClassifier(criterion = 'gini', random_state = 100, max_depth=3, min_samples_leaf=5)
Titanic_gini.fit(X_train, y_train)

#Predicting the outcome variable using the subdata X_test
y_prediction = Titanic_gini.predict(X_test)
y_prediction 

y_predictionDF = pd.DataFrame(data = y_prediction, columns = ['survived'])
y_predictionDF

RightDF = X_test
RightDF

y_DF = RightDF.merge(y_predictionDF, left_on = 'PassengerId' , right_on = 'survived')

selection = ['PassengerId', 'survived']
y_DF = y_DF[selection]
y_DF

#Accuracy of our model: the ratio of the predicted outcome to all the predicted data point
print('Accuracy is:')
accuracy = accuracy_score(y_test,y_prediction)*100
print(round(accuracy))


# Apply the trained model to a reallife test model
#import

Titanic_test_db = pd.read_csv('C:/Users/Seun/Desktop/Py_ML_Data/Titanic/test.csv')


features = ['Pclass','Fare', 'Parch']
test_features = Titanic_test_db[features]

#Defining predictor variable on the test data
X_Titanic_test_db = test_features
print(X_Titanic_test_db)

X_Titanic_test_db.head()

#working on the data to remove none or missing data
X_Titanic_test = X_Titanic_test_db.fillna(0)

pd.isna(X_Titanic_test)

#Predicting the outcome variable using real life test data
y_Titanic_test_prediction = Titanic_gini.predict(X_Titanic_test)
y_Titanic_test_prediction

